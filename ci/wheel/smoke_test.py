# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke test for the trimmed gnm-shape wheel built by build-wheel.yml.

Verifies, after installing the freshly built wheel, that:
  1. The NumPy backend imports without any heavy framework (no TF/JAX/torch).
  2. The bundled v3 head model data loads from the installed package.
  3. A zero-parameter evaluation reproduces the template mesh.

Run with the wheel installed:  python ci/wheel/smoke_test.py
"""

import sys

import numpy as np

EXPECTED_NUM_VERTICES = 17821
EXPECTED_IDENTITY_DIM = 253
EXPECTED_EXPRESSION_DIM = 383
EXPECTED_NUM_JOINTS = 4


def main() -> int:
  # Guard: none of the heavy frameworks may be required to import gnm_numpy.
  for banned in ('tensorflow', 'torch', 'jax', 'cv2', 'trimesh'):
    if banned in sys.modules:
      raise AssertionError(f'{banned} must not be imported by the core wheel')

  from gnm.shape import gnm_numpy

  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )

  assert gnm.num_vertices == EXPECTED_NUM_VERTICES, gnm.num_vertices
  assert gnm.identity_dim == EXPECTED_IDENTITY_DIM, gnm.identity_dim
  assert gnm.expression_dim == EXPECTED_EXPRESSION_DIM, gnm.expression_dim
  assert gnm.num_joints == EXPECTED_NUM_JOINTS, gnm.num_joints
  assert len(gnm.vertex_group_names) == 46, len(gnm.vertex_group_names)

  identity = np.zeros(gnm.identity_dim)
  expression = np.zeros(gnm.expression_dim)
  rotations = np.zeros((gnm.num_joints, 3))
  translation = np.zeros(3)

  vertices = gnm(identity, expression, rotations, translation)
  assert vertices.shape == (EXPECTED_NUM_VERTICES, 3), vertices.shape
  assert np.allclose(
      vertices, gnm.template_vertex_positions, atol=1e-5
  ), 'zero-parameter evaluation must reproduce the template mesh'

  for banned in ('tensorflow', 'torch', 'jax', 'cv2', 'trimesh'):
    if banned in sys.modules:
      raise AssertionError(f'{banned} must not be imported by the core wheel')

  print(
      'SMOKE TEST OK:',
      f'vertices={vertices.shape}, identity_dim={gnm.identity_dim},',
      f'expression_dim={gnm.expression_dim}, joints={gnm.num_joints}',
  )
  return 0


if __name__ == '__main__':
  sys.exit(main())
