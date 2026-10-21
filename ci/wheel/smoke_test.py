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
  2. The bundled v3 head model data ships in the wheel and loads *offline*
     (from_local must not fall back to the from_remote download path).
  3. A zero-parameter evaluation reproduces the template mesh.
  4. The pose helpers used by the AudioRig add-on (bind pose + pose
     correctives) run and match the template at rest.
  5. The new UDIM accessors work with the bundled (pre-UDIM) model file.

Run with the wheel installed:  python ci/wheel/smoke_test.py
"""

import sys

import numpy as np

EXPECTED_NUM_VERTICES = 17821
EXPECTED_IDENTITY_DIM = 253
EXPECTED_EXPRESSION_DIM = 383
EXPECTED_NUM_JOINTS = 4
EXPECTED_NUM_VERTEX_GROUPS = 46

BANNED_MODULES = ('tensorflow', 'torch', 'jax', 'cv2', 'trimesh')


def _assert_no_heavy_frameworks() -> None:
  for banned in BANNED_MODULES:
    if banned in sys.modules:
      raise AssertionError(f'{banned} must not be imported by the core wheel')


def main() -> int:
  _assert_no_heavy_frameworks()

  from gnm.shape import gnm_data_loader
  from gnm.shape import gnm_numpy

  # 1. The model file must be packaged inside the wheel, not downloaded.
  model_path = gnm_data_loader.get_packaged_model_path(
      gnm_numpy.GNMMajorVersion.V3, gnm_numpy.GNMVariant.HEAD
  )
  if not model_path.is_file():
    raise AssertionError(f'bundled model file is missing: {model_path}')

  # 2. from_local must not touch the network: break the download path.
  def _no_download(*args, **kwargs):
    raise AssertionError(
        'from_local() tried to download the model; the wheel must ship it'
    )

  original_remote_loader = gnm_data_loader.load_model_from_remote
  gnm_data_loader.load_model_from_remote = _no_download
  try:
    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )
  finally:
    gnm_data_loader.load_model_from_remote = original_remote_loader

  assert gnm.num_vertices == EXPECTED_NUM_VERTICES, gnm.num_vertices
  assert gnm.identity_dim == EXPECTED_IDENTITY_DIM, gnm.identity_dim
  assert gnm.expression_dim == EXPECTED_EXPRESSION_DIM, gnm.expression_dim
  assert gnm.num_joints == EXPECTED_NUM_JOINTS, gnm.num_joints
  assert len(gnm.vertex_group_names) == EXPECTED_NUM_VERTEX_GROUPS, len(
      gnm.vertex_group_names
  )

  identity = np.zeros(gnm.identity_dim)
  expression = np.zeros(gnm.expression_dim)
  rotations = np.zeros((gnm.num_joints, 3))
  translation = np.zeros(3)

  # 3. Zero-parameter evaluation reproduces the template mesh.
  vertices = gnm(identity, expression, rotations, translation)
  assert vertices.shape == (EXPECTED_NUM_VERTICES, 3), vertices.shape
  assert np.allclose(
      vertices, gnm.template_vertex_positions, atol=1e-5
  ), 'zero-parameter evaluation must reproduce the template mesh'

  # 4. AudioRig path: bind pose + pose correctives, without the rigid rotation.
  bind_pose = np.asarray(gnm.vertex_positions_bind_pose(identity, expression))
  correctives = np.asarray(gnm.compute_pose_correctives(rotations))
  assert bind_pose.shape == (EXPECTED_NUM_VERTICES, 3), bind_pose.shape
  assert correctives.shape == (EXPECTED_NUM_VERTICES, 3), correctives.shape
  assert np.allclose(
      bind_pose + correctives, gnm.template_vertex_positions, atol=1e-5
  ), 'bind pose + zero correctives must reproduce the template mesh'

  # 5. UDIM accessors (added after the bundled model was generated) are
  # computed on access, so they must work with this model file too.
  assert gnm.quad_uvs.shape == (gnm.quads.shape[0], 4, 2), gnm.quad_uvs.shape
  assert gnm.triangle_uvs.shape == (
      gnm.triangles.shape[0],
      3,
      2,
  ), gnm.triangle_uvs.shape
  assert gnm.quad_uvs_udim.shape == gnm.quad_uvs.shape
  assert gnm.triangle_uvs_udim.shape == gnm.triangle_uvs.shape
  assert gnm.quad_udim_tiles.shape == (gnm.quads.shape[0],)
  assert gnm.triangle_udim_tiles.shape == (gnm.triangles.shape[0],)

  _assert_no_heavy_frameworks()

  print(
      'SMOKE TEST OK:',
      f'vertices={vertices.shape}, identity_dim={gnm.identity_dim},',
      f'expression_dim={gnm.expression_dim}, joints={gnm.num_joints},',
      f'vertex_groups={len(gnm.vertex_group_names)},',
      f'udim_tiles={sorted(set(gnm.quad_udim_tiles.tolist()))}',
  )
  return 0


if __name__ == '__main__':
  sys.exit(main())
