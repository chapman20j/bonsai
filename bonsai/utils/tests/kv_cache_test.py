# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from bonsai.utils.kv_cache import CyclicCache, LayerCache


class TestLayerCache(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.cache_size = 10
        self.num_kv_heads = 1
        self.head_dim = 1
        self.dtype = jnp.float32

        self.lc = LayerCache(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            batch_size=self.batch_size,
            cache_size=self.cache_size,
            dtype=self.dtype,
        )

    def test_init(self):
        self.assertEqual(self.lc.k_cache.shape, (2, 10, 1, 1))
        self.assertEqual(self.lc.cur_ind[...], 0)
        self.assertTrue(jnp.all(self.lc.start_ind[...] == -1))

    def test_prefill(self):
        segment_ids = jnp.array([[0, 0, 1, 1], [1, 1, 1, 1]])
        k_new = jnp.ones((2, 4, 1, 1))
        v_new = jnp.ones((2, 4, 1, 1))

        self.lc._init_start_ind(segment_ids)
        self.lc.prefill(k_new, v_new, segment_ids)

        self.assertEqual(self.lc.cur_ind[...], 4)
        np.testing.assert_array_equal(self.lc.start_ind[...], np.array([2, 0]))
        np.testing.assert_array_equal(self.lc.k_cache[:, :4, :, :], k_new)

    def test_update(self):
        segment_ids = jnp.array([[1], [1]])
        self.lc._init_start_ind(segment_ids)
        k_val = jnp.array([[[[10.0]]], [[[20.0]]]])
        v_val = jnp.array([[[[10.0]]], [[[20.0]]]])

        self.lc.update(k_val, v_val)
        self.assertEqual(self.lc.cur_ind[...], 1)
        self.assertEqual(self.lc.k_cache[0, 0, 0, 0], 10.0)
        self.assertEqual(self.lc.k_cache[1, 0, 0, 0], 20.0)

    def test_compute_causal_mask(self):
        segment_ids = jnp.array([[0, 1], [1, 1]])
        self.lc._init_start_ind(segment_ids)
        mask = self.lc.compute_causal_mask(2)

        self.assertFalse(jnp.any(mask[0, 0, :]))
        self.assertTrue(mask[1, 0, 0])
        self.assertFalse(mask[1, 0, 1])


class TestCyclicCache(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 1
        self.cache_size = 4
        self.num_kv_heads = 1
        self.head_dim = 1
        self.dtype = jnp.float32

        self.cc = CyclicCache(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            batch_size=self.batch_size,
            cache_size=self.cache_size,
            dtype=self.dtype,
        )

    def test_init(self):
        self.assertEqual(self.cc.k_cache.shape, (1, 4, 1, 1))

    def test_prefill(self):
        segment_ids = jnp.array([[1, 1]])
        k = jnp.ones((1, 2, 1, 1))
        self.cc._init_start_ind(segment_ids)
        self.cc.prefill(k, k, segment_ids)
        self.assertEqual(self.cc.cur_ind[...], 2)
        np.testing.assert_array_equal(self.cc.k_cache[:, :2, :, :], k)

    def test_update_after_cache_full(self):
        segment_ids = jnp.ones((1, 4), dtype=jnp.int32)
        self.cc._init_start_ind(segment_ids)

        k_fill = jnp.ones((1, 4, 1, 1))
        self.cc.update(k_fill, k_fill)

        k_new = jnp.array([[[[99.0]]]])
        self.cc.update(k_new, k_new)

        self.assertEqual(self.cc.cur_ind[...], 5)
        self.assertEqual(self.cc.k_cache[0, 0, 0, 0], 99.0)
        self.assertEqual(self.cc.k_cache[0, 1, 0, 0], 1.0)

    def test_compute_causal_mask(self):
        segment_ids = jnp.ones((1, 5), dtype=jnp.int32)
        self.cc._init_start_ind(segment_ids)
        self.cc.update(jnp.ones((1, 4, 1, 1)), jnp.ones((1, 4, 1, 1)))
        self.cc.update(jnp.ones((1, 1, 1, 1)) * 2, jnp.ones((1, 1, 1, 1)) * 2)
        mask = self.cc.compute_causal_mask(1)
        self.assertTrue(jnp.all(mask[0, 0, :]))

    def test_sliding_window_boundary(self):
        segment_ids = jnp.ones((1, 1), dtype=jnp.int32)
        self.cc._init_start_ind(segment_ids)
        self.cc.cur_ind[...] = 6

        mask = self.cc.compute_causal_mask(1)  # Mask for token at Pos 5
        self.assertTrue(jnp.all(mask[0, 0, :]))

        self.cc.cur_ind[...] = 10
        mask_large = self.cc.compute_causal_mask(1)
        self.assertTrue(jnp.all(mask_large[0, 0, :]))


if __name__ == "__main__":
    absltest.main()
