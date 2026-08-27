#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import MagicMock, patch

from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.launcher.api import launch_agent, LaunchConfig
from torch.testing._internal.common_utils import run_tests, TestCase


class LauncherApiTest(TestCase):
    def setUp(self):
        super().setUp()
        # Save original environment variable if it exists
        self.original_signals_env = os.environ.get(
            "TORCHELASTIC_SIGNALS_TO_HANDLE", None
        )

    def tearDown(self):
        # Restore original environment variable
        if self.original_signals_env is not None:
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = self.original_signals_env
        elif "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_launch_agent_sets_signals_env_var(self, mock_get_handler, mock_agent):
        """Test that launch_agent sets the TORCHELASTIC_SIGNALS_TO_HANDLE environment variable."""
        # Setup
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",
        )
        entrypoint = "dummy_script.py"
        args = []

        # Make sure the environment variable doesn't exist before the test
        if "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

        # Mock agent.run() to return a MagicMock
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = MagicMock(
            is_failed=lambda: False, return_values={}
        )
        mock_agent.return_value = mock_agent_instance

        # Call launch_agent
        launch_agent(config, entrypoint, args)

        # Verify that the environment variable was set correctly
        self.assertEqual(
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"], "SIGTERM,SIGUSR1,SIGUSR2"
        )

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_launch_agent_default_signals(self, mock_get_handler, mock_agent):
        """Test that launch_agent uses the default signals if not specified."""
        # Setup
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            # Not specifying signals_to_handle, should use default
        )
        entrypoint = "dummy_script.py"
        args = []

        # Make sure the environment variable doesn't exist before the test
        if "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

        # Mock agent.run() to return a MagicMock
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = MagicMock(
            is_failed=lambda: False, return_values={}
        )
        mock_agent.return_value = mock_agent_instance

        # Call launch_agent
        launch_agent(config, entrypoint, args)

        # Verify that the environment variable was set to the default value
        self.assertEqual(
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"],
            "SIGTERM,SIGINT,SIGHUP,SIGQUIT",
        )

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_rdzv_shutdown_failure_does_not_mask_child_failure(
        self, mock_get_handler, mock_agent
    ):
        """A failing rendezvous shutdown must not replace the worker failure.

        ChildFailedError is what names the worker that actually died, and it
        used to be discarded when shutdown() raised from the finally block.
        """
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)

        handler = MagicMock()
        handler.shutdown.side_effect = RuntimeError("rendezvous store unreachable")
        mock_get_handler.return_value = handler

        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = MagicMock(
            is_failed=lambda: True, failures={}
        )
        mock_agent.return_value = mock_agent_instance

        with self.assertRaises(ChildFailedError):
            launch_agent(config, "dummy_script.py", [])

        handler.shutdown.assert_called_once()

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_rdzv_shutdown_failure_does_not_mask_other_errors(
        self, mock_get_handler, mock_agent
    ):
        """The same holds for any other error already on its way out."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)

        handler = MagicMock()
        handler.shutdown.side_effect = RuntimeError("rendezvous store unreachable")
        mock_get_handler.return_value = handler

        mock_agent_instance = MagicMock()
        mock_agent_instance.run.side_effect = ValueError("the real problem")
        mock_agent.return_value = mock_agent_instance

        with self.assertRaises(ValueError):
            launch_agent(config, "dummy_script.py", [])

        handler.shutdown.assert_called_once()

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_rdzv_shutdown_failure_on_successful_run_is_logged_not_raised(
        self, mock_get_handler, mock_agent
    ):
        """With nothing in flight, a failing shutdown is reported, not raised.

        The run itself succeeded, so its return values are still handed back.
        """
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)

        handler = MagicMock()
        handler.shutdown.side_effect = RuntimeError("rendezvous store unreachable")
        mock_get_handler.return_value = handler

        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = MagicMock(
            is_failed=lambda: False, return_values={0: "result"}
        )
        mock_agent.return_value = mock_agent_instance

        with self.assertLogs(
            "torch.distributed.launcher.api", level="ERROR"
        ) as captured:
            result = launch_agent(config, "dummy_script.py", [])

        self.assertEqual({0: "result"}, result)
        handler.shutdown.assert_called_once()
        self.assertTrue(
            any("rendezvous store unreachable" in line for line in captured.output),
            f"expected the shutdown failure in the log, got {captured.output}",
        )


if __name__ == "__main__":
    run_tests()
