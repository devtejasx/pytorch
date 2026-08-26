# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import socket
import sys
import unittest
from unittest.mock import patch

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer, find_free_port


if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)


class EtcdServerTest(unittest.TestCase):
    def test_etcd_server_start_stop(self):
        server = EtcdServer()
        server.start()

        try:
            port = server.get_port()
            host = server.get_host()

            self.assertGreater(port, 0)
            self.assertEqual("localhost", host)
            self.assertEqual(f"{host}:{port}", server.get_endpoint())
            self.assertIsNotNone(server.get_client().version)
        finally:
            server.stop()

    def test_etcd_server_with_rendezvous(self):
        server = EtcdServer()
        server.start()

        try:
            endpoint = server.get_endpoint()
            rdzv_params = RendezvousParameters(
                backend="etcd",
                endpoint=endpoint,
                run_id="test_run_1",
                min_nodes=1,
                max_nodes=1,
                timeout=60,
                last_call_timeout=30,
                local_addr="127.0.0.1",
            )
            rdzv_handler = create_rdzv_handler(rdzv_params)
            rdzv_info = rdzv_handler.next_rendezvous()
            self.assertIsNotNone(rdzv_info.store)
            self.assertEqual(0, rdzv_info.rank)
            self.assertEqual(1, rdzv_info.world_size)
        finally:
            server.stop()


class FindFreePortTest(unittest.TestCase):
    def test_socket_creation_failure_is_not_masked(self):
        # socket() raising means no socket was created, so there is nothing to
        # close. Closing regardless raised UnboundLocalError and hid the real
        # error.
        with (
            patch(
                "socket.getaddrinfo",
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
                ],
            ),
            patch("socket.socket", side_effect=OSError("socket failed")),
        ):
            with self.assertRaises(RuntimeError):
                find_free_port()

    def test_remaining_addresses_are_tried(self):
        real_socket = socket.socket
        attempts = []

        def flaky_socket(family, type, proto):
            attempts.append((family, type, proto))
            if len(attempts) == 1:
                raise OSError("socket failed on first address")
            return real_socket(socket.AF_INET, socket.SOCK_STREAM, 0)

        with (
            patch(
                "socket.getaddrinfo",
                return_value=[
                    (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0)),
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
                ],
            ),
            patch("socket.socket", side_effect=flaky_socket),
        ):
            sock = find_free_port()

        try:
            self.assertEqual(2, len(attempts))
            self.assertGreater(sock.getsockname()[1], 0)
        finally:
            sock.close()

    def test_created_socket_is_closed_when_bind_fails(self):
        closed = []

        class FakeSocket:
            def bind(self, addr):
                raise OSError("bind failed")

            def close(self):
                closed.append(self)

        with (
            patch(
                "socket.getaddrinfo",
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
                ],
            ),
            patch("socket.socket", return_value=FakeSocket()),
        ):
            with self.assertRaises(RuntimeError):
                find_free_port()

        self.assertEqual(1, len(closed))
