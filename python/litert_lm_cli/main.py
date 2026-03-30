# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script for litert-lm binary."""

import datetime
import os
import shutil
import subprocess

import click

import litert_lm
from litert_lm_cli import help_formatter
from litert_lm_cli import model
from litert_lm_cli import venv_manager
from litert_lm_cli import version


@click.group(
    cls=help_formatter.ColorGroup,
    name="litert-lm",
    context_settings=dict(show_default=True, max_content_width=120),
)
@click.version_option(version=version.VERSION)
def cli():
  """CLI tool for LiteRT-LM models."""


@cli.command(
    context_settings=dict(ignore_unknown_options=True),
    help="""Converts a HuggingFace model to LiteRT-LM format.

  SOURCE: The HuggingFace model ID or path (e.g., "google/gemma-3-1b-it").

  The conversion process requires the `litert-torch` tool. These dependencies
  are optional and may not be supported on all platforms (e.g., Raspberry Pi).
  By default, `litert-lm` manages these dependencies in a standalone virtual
  environment and installs them on-demand to avoid conflicts with your
  environment. If you prefer using the current active venv, run with
  `--prefer_current_venv`.""",
)
@click.argument("source")
@click.option(
    "--model_id",
    default=None,
    help="The ID to store the model as. Defaults to source.",
)
@click.option(
    "--prefer_current_venv",
    is_flag=True,
    default=False,
    help="Whether to use the currently active virtual environment.",
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def convert(source, model_id=None, prefer_current_venv=False, extra_args=()):
  """Converts a HuggingFace model to LiteRT-LM format.

  Args:
    source: The HuggingFace model ID or path (e.g., "google/gemma-3-1b-it").
    model_id: The ID to store the model as. Defaults to source.
    prefer_current_venv: Whether to use the currently active virtual
      environment.
    extra_args: Additional arguments passed to litert-torch.
  """
  effective_model_id = model_id or source
  if any(
      m.model_id == effective_model_id for m in model.Model.get_all_models()
  ):
    click.echo(
        click.style(
            f"Error: Model ID '{effective_model_id}' already exists.\n",
            fg="red",
            bold=True,
        )
    )
    click.echo("Suggestions:")
    click.echo(
        "  1. Run the existing model with 'litert-lm run"
        f" {effective_model_id}'."
    )
    click.echo(
        f"  2. Convert again using 'litert-lm convert {effective_model_id}'"
        " with '--model_id=other-model-id' to set a different model ID for"
        " the converted model."
    )
    click.echo(
        "  3. Rename the existing model with 'litert-lm rename"
        f" {effective_model_id} <new_model_id>' and convert the model again."
    )
    return

  vm = venv_manager.VenvManager(prefer_current_venv=prefer_current_venv)
  vm.recreate_venv_if_self_managed()
  vm.ensure_binary(vm.litert_torch_bin)

  output_dir = model.get_model_dir(effective_model_id)
  os.makedirs(output_dir, exist_ok=True)

  cmd = [
      vm.litert_torch_bin,
      "export_hf",
      "--model",
      source,
      "--output_dir",
      output_dir,
      "--bundle_litert_lm",
  ]

  cmd.extend(extra_args)

  click.echo(click.style(f"Running: {' '.join(cmd)}", fg="cyan"))
  try:
    subprocess.run(cmd, check=True)
  except subprocess.CalledProcessError as e:
    click.echo(
        click.style(
            f"Error: Model conversion failed with exit code {e.returncode}.",
            fg="red",
            bold=True,
        )
    )
    click.echo("Check the logs above for the specific error message.")
    return

  click.echo(
      click.style(
          f"You can now run the model with 'run {effective_model_id}'",
          fg="green",
      )
  )


@cli.command(name="list")
def list_models():
  """Lists all imported LiteRT-LM models."""
  base_dir = model.get_converted_models_base_dir()
  click.echo(f"Listing models in: {base_dir}")

  models = sorted(model.Model.get_all_models(), key=lambda m: m.model_id)

  # Calculate dynamic width for ID column
  id_width = max([len(m.model_id) for m in models] + [len("ID"), 25]) + 2

  click.echo(
      click.style(f"{'ID':<{id_width}} {'SIZE':<15} {'MODIFIED'}", bold=True)
  )

  for model_item in models:
    path = model_item.model_path
    try:
      stat = os.stat(path)
      size_bytes = stat.st_size
      if size_bytes >= 1024 * 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
      else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
      modified_date = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
          "%Y-%m-%d %H:%M:%S"
      )
    except FileNotFoundError:
      size_str = "Unknown"
      modified_date = "Unknown"

    click.echo(
        f"{model_item.model_id:<{id_width}} {size_str:<15} {modified_date}"
    )


@cli.command(name="import")
@click.argument("source")
@click.argument("ref")
def import_model(source, ref):
  """Imports a model from a local path."""
  if not os.path.exists(source):
    click.echo(click.style(f"Source file not found: {source}", fg="red"))
    return

  model_obj = model.Model.from_model_id(ref)
  model_path = model_obj.model_path
  model_dir = os.path.dirname(model_path)

  os.makedirs(model_dir, exist_ok=True)

  shutil.copy(source, model_path)
  click.echo(
      click.style(f"Successfully imported model to {model_path}", fg="green")
  )


@cli.command(help="Deletes a model from the local storage.")
@click.argument("model_id")
def delete(model_id):
  """Deletes a model from the local storage.

  Args:
    model_id: The ID of the model to delete.
  """
  model_obj = model.Model.from_model_id(model_id)
  model_dir = os.path.dirname(model_obj.model_path)
  if os.path.exists(model_dir) and model_dir.startswith(
      model.get_converted_models_base_dir()
  ):
    shutil.rmtree(model_dir)
    click.echo(click.style(f"Deleted model: {model_id}", fg="green"))
  else:
    click.echo(click.style(f"Model not found: {model_id}", fg="red"))


@cli.command(help="Renames a model.")
@click.argument("old_model_id")
@click.argument("new_model_id")
def rename(old_model_id, new_model_id):
  """Renames a model.

  Args:
    old_model_id: The current model ID.
    new_model_id: The new model ID.
  """
  old_model = model.Model.from_model_id(old_model_id)
  if not old_model.exists():
    click.echo(click.style(f"Model not found: {old_model_id}", fg="red"))
    return

  new_model = model.Model.from_model_id(new_model_id)
  if new_model.exists():
    click.echo(
        click.style(f"Target model ID already exists: {new_model_id}", fg="red")
    )
    return

  old_dir = os.path.dirname(old_model.model_path)
  new_dir = os.path.dirname(new_model.model_path)

  os.makedirs(os.path.dirname(new_dir), exist_ok=True)
  shutil.move(old_dir, new_dir)
  click.echo(
      click.style(
          f'Renamed model "{old_model_id}" to "{new_model_id}"', fg="green"
      )
  )


def parse_speculative_decoding(unused_ctx, unused_param, value):
  """Click callback to parse speculative decoding mode strings into bool | None.

  Args:
    unused_ctx: The click context.
    unused_param: The click parameter.
    value: The value to parse ("auto", "true", or "false").

  Returns:
    True for "true", False for "false", and None for "auto".
  """
  if value is None:
    return None
  value_lower = value.lower()
  if value_lower == "auto":
    return None
  elif value_lower == "true":
    return True
  elif value_lower == "false":
    return False
  return value


def common_inference_options(f):
  """Decorator for common options shared across commands."""
  f = click.option(
      "--verbose",
      is_flag=True,
      default=False,
      help="Whether to enable verbose logging.",
  )(f)
  f = click.option(
      "--enable-speculative-decoding",
      type=click.Choice(["auto", "true", "false"], case_sensitive=False),
      default="auto",
      callback=parse_speculative_decoding,
      help="""\b
Speculative decoding mode ("auto", "true", "false").
  - auto: Automatically determine the speculative decoding behavior from the model metadata.
  - true: Force enable speculative decoding. It will throw an error if the model does not support it.
  - false: Force disable speculative decoding.
""",
  )(f)
  f = click.option(
      "-b",
      "--backend",
      type=click.Choice(["cpu", "gpu"], case_sensitive=False),
      default="cpu",
      help="The backend to use.",
  )(f)
  return f


@cli.command(help="Benchmarks a LiteRT-LM model.")
@click.argument("model_reference")
@click.option(
    "-p",
    "--prefill_tokens",
    default=256,
    type=int,
    help="The number of tokens to prefill.",
)
@click.option(
    "-d",
    "--decode_tokens",
    default=256,
    type=int,
    help="The number of tokens to decode.",
)
@common_inference_options
def benchmark(
    model_reference: str,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    backend: str = "cpu",
    android: bool = False,
    enable_speculative_decoding: bool | None = None,
    verbose: bool = False,
):
  """Benchmarks a LiteRT-LM model.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`.
    prefill_tokens: The number of tokens to prefill.
    decode_tokens: The number of tokens to decode.
    backend: The backend to use (cpu or gpu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  model_obj = model.Model.from_model_reference(model_reference)
  model_obj.benchmark(
      prefill_tokens=prefill_tokens,
      decode_tokens=decode_tokens,
      is_android=android,
      backend=backend,
      enable_speculative_decoding=enable_speculative_decoding,
  )


@cli.command(
    help=(
        "\b\n"
        "Runs a LiteRT-LM model interactively or with a single prompt.\n"
        "\n"
        "\b\n"
        "Example preset file:\n"
        "  ```py\n"
        "  def add_numbers(a: float, b: float) -> float:\n"
        "    '''Adds two numbers.'''\n"
        "    return a + b\n"
        "\n"
        "\b\n"
        "  # Provides the 'system instruction', 'tools', and 'extra_context'\n"
        "  system_instruction = 'You are a helpful assistant.'\n"
        "  tools = [add_numbers]\n"
        "  extra_context = {'key': 'value'}\n"
        "  ```"
    )
)
@click.argument("model_reference")
@click.option(
    "--prompt", default=None, help="A single prompt to run once and exit."
)
@click.option(
    "--preset",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to a Python file containing tool functions and system"
        " instructions."
    ),
)
@common_inference_options
def run(
    model_reference,
    prompt=None,
    preset=None,
    backend="cpu",
    android=False,
    enable_speculative_decoding=None,
    verbose=False,
):
  r"""Runs a LiteRT-LM model interactively or with a single prompt.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`.
    prompt: A single prompt to run once and exit.
    preset: Path to a Python file containing tool functions and system
      instructions.
    backend: The backend to use (cpu or gpu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  model_obj = model.Model.from_model_reference(model_reference)
  if not model_obj.exists():
    # Only auto-convert if it looks like a HuggingFace repo ID (account/repo)
    # and is not a local path.
    parts = model_reference.split("/")
    if len(parts) == 2 and all(parts) and not os.path.exists(model_reference):
      click.echo(
          click.style(
              f"Model '{model_reference}' not found. Attempting to convert"
              f" from https://huggingface.co/{model_reference} ...",
              fg="yellow",
          )
      )
      convert.callback(source=model_reference)
      model_obj = model.Model.from_model_reference(model_reference)

    if not model_obj.exists():
      click.echo(
          click.style(
              f"Failed to find or convert model '{model_reference}'.", fg="red"
          )
      )
      return

  model_obj.run_interactive(
      prompt=prompt,
      is_android=android,
      backend=backend,
      preset=preset,
      enable_speculative_decoding=enable_speculative_decoding,
  )


def main():
  litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
  cli()


if __name__ == "__main__":
  main()
