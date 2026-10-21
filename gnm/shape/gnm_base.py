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

"""Base GNM class."""

from __future__ import annotations

import abc
from collections.abc import Mapping
import dataclasses
from typing import Any, Self

from etils import epath
from gnm.shape import gnm_data_loader
from gnm.shape import gnm_factory_mixins
from gnm.shape.data.versions import gnm_specs



# `from_remote`) into separate public/internal mixin modules, selected via
# copybara, to reduce the number of copybara strips in this class.
@dataclasses.dataclass(init=False)
class GNMBase(gnm_factory_mixins.GNMFactoryMethodsMixin, abc.ABC):
  """Base GNM class."""

  version: gnm_specs.GNMVersion
  variant: gnm_specs.GNMVariant

  @classmethod
  def from_local(
      cls,
      version: gnm_specs.GNMMajorVersion,
      variant: gnm_specs.GNMVariant,
  ) -> Self:
    """Creates a GNM instance from the model file bundled in this wheel.

    Fork of google/GNM: upstream deprecated `from_local()` in favor of the
    `from_remote()` download helpers and removed the model files from the
    package, but the AudioRig wheel ships `data/**/*.npz` so that Blender can
    load the model fully offline. The bundled file is read through
    `from_custom_file()`, so it goes through the same validation as any custom
    model file.

    Args:
      version: GNM major version.
      variant: GNM model variant.

    Returns:
      A GNM instance loaded with the model weights.
    """
    return cls.from_custom_file(
        gnm_data_loader.get_packaged_model_path(version, variant)
    )

  @classmethod
  def from_remote(
      cls,
      version: gnm_specs.GNMMajorVersion,
      variant: gnm_specs.GNMVariant,
      source: gnm_specs.GNMRemoteSource = gnm_specs.GNMRemoteSource.HTTP,
      *,
      cache_dir: epath.PathLike | None = None,
      force_download: bool = False,
  ) -> Self:
    """Creates a GNM instance from a remote repository.

    Args:
      version: GNM major version.
      variant: GNM model variant.
      source: Remote repository source (HTTP, Hugging Face, or Kaggle).
      cache_dir: Optional custom directory (Path or str) to cache downloaded
        models.
      force_download: If True, forces redownload even if cached locally.

    Returns:
      A GNM instance loaded with the model weights.
    """
    data_dict = gnm_data_loader.load_model_from_remote(
        version=version,
        variant=variant,
        source=source,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    return cls._from_model_data(data_dict)

  @classmethod
  def from_gnm(cls, gnm: GNMBase) -> Self:
    """Creates a GNM instance from another GNM instance."""
    data_dict = gnm.to_numpy_data_dict()
    return cls._from_model_data(data_dict)

  @abc.abstractmethod
  def to_numpy_data_dict(self) -> dict[str, Any]:
    """Returns a dictionary of the GNM data represented as NumPy arrays."""
    raise NotImplementedError()

  @classmethod
  @abc.abstractmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> Self:
    """Creates a GNM instance from a model data."""
    pass

  @property
  def major_version(self) -> gnm_specs.GNMMajorVersion:
    """Returns the major version of the model."""
    return gnm_data_loader.full_version_to_major(self.version)

  @property
  def body_part(self) -> gnm_specs.GNMBodyPart:
    """Returns the body part of the GNM model."""
    return gnm_specs.GNM_VARIANT_TO_BODY_PART_MAP[self.variant]
