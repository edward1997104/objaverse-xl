"""Script to download objects from GitHub."""

import json
import multiprocessing
import pickle
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import random
from multiprocessing import Pool
from typing import Callable, Dict, List, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

import fsspec
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from objaverse.utils import get_file_hash
from objaverse.xl.abstract import ObjaverseSource

FILE_EXTENSIONS = [
    ".obj",
    ".glb",
    ".gltf",
    ".usdz",
    ".usd",
    ".fbx",
    ".stl",
    ".usda",
    ".dae",
    ".ply",
    ".abc",
    ".blend",
]

class GitHubDownloader(ObjaverseSource):
    """Script to download objects from GitHub."""

    _uuid_mappings_cache: Optional[Dict[str, str]] = None
    _uuid_mappings_cache_path: Optional[str] = None

    @classmethod
    def _get_annotations(
        cls, url: str, filename: str, download_dir: str, refresh: bool
    ) -> pd.DataFrame:
        """Load the annotations from a given file URL.

        Args:
            url (str): The URL to the annotations file.
            filename (str): The filename of the annotations file.
            download_dir (str): The directory to load the annotations from.
                Supports all file systems supported by fsspec.
            refresh (bool): Whether to refresh the annotations by downloading
                them from the remote source.

        Returns:
            pd.DataFrame: The annotations, which includes the columns "thingId", "fileId",
                "filename", and "license".
        """
        filename = os.path.join(download_dir, "github", filename)
        fs, path = fsspec.core.url_to_fs(filename)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        # download the parquet file if it doesn't exist
        if refresh or not fs.exists(path):
            logger.info(f"Downloading {url} to {filename}")
            response = requests.get(url)
            response.raise_for_status()
            with fs.open(path, "wb") as file:
                file.write(response.content)

        # load the parquet file with fsspec
        with fs.open(path) as f:
            df = pd.read_parquet(f)

        return df

    @classmethod
    def get_annotations(
        cls, download_dir: str = "~/.objaverse", refresh: bool = False
    ) -> pd.DataFrame:
        """Loads the GitHub 3D object metadata as a Pandas DataFrame.

        Args:
            download_dir (str, optional): Directory to download the parquet metadata
                file. Supports all file systems supported by fsspec. Defaults to
                "~/.objaverse".
            refresh (bool, optional): Whether to refresh the annotations by downloading
                them from the remote source. Defaults to False.

        Returns:
            pd.DataFrame: GitHub 3D object metadata as a Pandas DataFrame with columns
                for the object "fileIdentifier", "license", "source", "fileType",
                "sha256", and "metadata".
        """
        return cls._get_annotations(
            url="https://huggingface.co/datasets/allenai/objaverse-xl/resolve/main/github/github.parquet",
            filename="github.parquet",
            download_dir=download_dir,
            refresh=refresh,
        )

    @classmethod
    def get_alignment_annotations(
        cls, download_dir: str = "~/.objaverse", refresh: bool = False
    ) -> pd.DataFrame:
        """Loads the alignment fine-tuning metadata as a Pandas DataFrame.

        Args:
            download_dir (str, optional): Directory to download the parquet metadata
                file. Supports all file systems supported by fsspec. Defaults to
                "~/.objaverse".
            refresh (bool, optional): Whether to refresh the annotations by downloading
                them from the remote source. Defaults to False.

        Returns:
            pd.DataFrame: GitHub 3D object metadata as a Pandas DataFrame with columns
                for the object "fileIdentifier", "license", "source", "fileType",
                "sha256", and "metadata".
        """
        return cls._get_annotations(
            url="https://huggingface.co/datasets/allenai/objaverse-xl/resolve/main/github/alignment.parquet",
            filename="alignment.parquet",
            download_dir=download_dir,
            refresh=refresh,
        )

    @classmethod
    def _get_repo_id_with_hash(cls, item: pd.Series) -> str:
        org, repo = item["fileIdentifier"].split("/")[3:5]
        commit_hash = item["fileIdentifier"].split("/")[6]
        return f"{org}/{repo}/{commit_hash}"

    @classmethod
    def _git_shallow_clone(cls, repo_url: str, target_directory: str) -> bool:
        """Helper function to shallow clone a repo with git.

        Args:
            repo_url (str): URL of the repo to clone.
            target_directory (str): Directory to clone the repo to.

        Returns:
            bool: True if the clone was successful, False otherwise.
        """
        clone_env = os.environ.copy()
        clone_env.setdefault("GIT_TERMINAL_PROMPT", "0")

        authenticated_repo_url = repo_url
        token = os.getenv("GITHUB_TOKEN")
        if token and repo_url.startswith(("https://", "http://")):
            parsed = urlsplit(repo_url)
            authenticated_repo_url = urlunsplit(
                (
                    parsed.scheme,
                    f"{token}@{parsed.netloc}",
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )

        command = ["git", "clone", "--depth", "1", authenticated_repo_url, target_directory]

        display_command = [
            "git",
            "clone",
            "--depth",
            "1",
            repo_url,
            target_directory,
        ]

        success = cls._run_command_with_check(
            command,
            env=clone_env,
            command_name=" ".join(display_command),
            max_retries=5,
            retry_delay=2.0,
        )

        if success and token:
            cls._run_command_with_check(
                ["git", "remote", "set-url", "origin", repo_url],
                cwd=target_directory,
                command_name="git remote set-url origin <redacted>",
            )

        return success

    @classmethod
    def _run_command_with_check(
        cls,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        command_name: Optional[str] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ) -> bool:
        """Helper function to run a command and check if it was successful.

        Args:
            command (List[str]): Command to run.
            cwd (Optional[str], optional): Current working directory to run the command
                in. Defaults to None.
            env (Optional[Dict[str, str]], optional): Environment variables to pass to
                the subprocess. Defaults to None.
            command_name (Optional[str], optional): Human readable command string to log
                instead of the raw command (useful for hiding secrets). Defaults to None.

        Returns:
            bool: True if the command was successful, False otherwise.
        """
        display = command_name if command_name is not None else " ".join(command)
        attempt = 0
        while True:
            try:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if completed.stdout:
                    logger.debug("stdout: {}", completed.stdout.strip())
                if completed.stderr:
                    logger.debug("stderr: {}", completed.stderr.strip())
                return True
            except subprocess.CalledProcessError as e:
                logger.error(
                    "Error running command `{}` (exit code {})", display, e.returncode
                )
                stdout_text = e.stdout.strip() if e.stdout else ""
                stderr_text = e.stderr.strip() if e.stderr else ""
                if stdout_text:
                    logger.error("stdout: {}", stdout_text)
                if stderr_text:
                    logger.error("stderr: {}", stderr_text)

                lowered = stderr_text.lower()
                auth_failure_keywords = (
                    "fatal: authentication failed",
                    "fatal: could not read password",
                    "permission denied (publickey)",
                    "remote: permission to",
                    "invalid username or password",
                )
                if any(keyword in lowered for keyword in auth_failure_keywords):
                    logger.warning(
                        "Authentication failure detected while running `{}`; not retrying.",
                        display,
                    )
                    return False

                if attempt < max_retries:
                    base_delay = retry_delay * (2**attempt)
                    attempt += 1
                    jitter = random.uniform(0.5, 1.5)
                    delay = base_delay * jitter
                    logger.warning(
                        "Retrying command `{}` in {:.1f}s (attempt {}/{})",
                        display,
                        delay,
                        attempt,
                        max_retries + 1,
                    )
                    time.sleep(delay)
                    continue

                return False

    @classmethod
    def _load_uuid_mappings(cls, mapping_path: str) -> Dict[str, str]:
        if (
            cls._uuid_mappings_cache is not None
            and cls._uuid_mappings_cache_path == mapping_path
        ):
            return cls._uuid_mappings_cache

        try:
            with open(mapping_path, "rb") as f:
                cls._uuid_mappings_cache = pickle.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Unable to load UUID mappings from {mapping_path}"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Error loading UUID mappings from {mapping_path}"
            ) from exc

        cls._uuid_mappings_cache_path = mapping_path
        return cls._uuid_mappings_cache

    @classmethod
    def _repo_has_existing_objects_on_s3(
        cls,
        repo_objects: Dict[str, str],
        uuid_mappings: Dict[str, str],
        path_exists_cache: Dict[str, bool],
    ) -> bool:
        for file_identifier in repo_objects:
            prefix = uuid_mappings.get(file_identifier)
            if not prefix:
                continue

            candidate_paths = {prefix}
            candidate_paths.update(prefix + ext for ext in FILE_EXTENSIONS)

            for candidate in candidate_paths:
                if candidate in path_exists_cache:
                    if path_exists_cache[candidate]:
                        return True
                    continue

                try:
                    fs, fs_path = fsspec.core.url_to_fs(candidate)
                except Exception as exc:  # pragma: no cover - requires remote fs
                    logger.debug(
                        "Skipping candidate {} due to filesystem error: {}", candidate, exc
                    )
                    path_exists_cache[candidate] = False
                    continue

                try:
                    exists = fs.exists(fs_path)
                except Exception as exc:  # pragma: no cover - requires remote fs
                    logger.debug(
                        "Skipping candidate {} due to existence check error: {}", candidate, exc
                    )
                    exists = False

                path_exists_cache[candidate] = exists
                if exists:
                    return True

        return False

    @classmethod
    def _process_repo(
        cls,
        repo_id: str,
        fs: fsspec.AbstractFileSystem,
        base_dir: str,
        save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]],
        expected_objects: Dict[str, str],
        handle_found_object: Optional[Callable],
        handle_modified_object: Optional[Callable],
        handle_missing_object: Optional[Callable],
        handle_new_object: Optional[Callable],
        commit_hash: Optional[str],
    ) -> Dict[str, str]:
        """Process a single repo.

        Args:
            repo_id (str): GitHub repo ID in the format of organization/repo.
            fs (fsspec.AbstractFileSystem): File system to use for saving the repo.
            base_dir (str): Base directory to save the repo to.
            expected_objects (Dict[str, str]): Dictionary of objects that one expects to
                find in the repo. Keys are the "fileIdentifier" (i.e., the GitHub URL in
                this case) and values are the "sha256" of the objects.
            {and the rest of the args are the same as download_objects}

        Returns:
            Dict[str, str]: A dictionary that maps from the "fileIdentifier" to the path
                of the downloaded object.
        """
        # NOTE: assuming that the user has already checked that the repo doesn't exist,
        org, repo = repo_id.split("/")

        out = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            # clone the repo to a temp directory
            target_directory = os.path.join(temp_dir, repo)
            successful_clone = cls._git_shallow_clone(
                f"https://github.com/{org}/{repo}.git", target_directory
            )
            if not successful_clone:
                logger.error(f"Could not clone {repo_id}")
                if handle_missing_object is not None:
                    for github_url, sha256 in expected_objects.items():
                        handle_missing_object(
                            file_identifier=github_url,
                            sha256=sha256,
                            metadata=dict(github_organization=org, github_repo=repo),
                        )
                return {}

            # use the commit hash if specified
            repo_commit_hash = cls._get_commit_hash_from_local_git_dir(target_directory)
            if commit_hash is not None:
                keep_going = True
                if repo_commit_hash != commit_hash:
                    # run git reset --hard && git checkout 37f4d8d287e201ce52c048bf74d46d6a09d26b2c
                    if not cls._run_command_with_check(
                        ["git", "fetch", "origin", commit_hash],
                        target_directory,
                        max_retries=5,
                        retry_delay=2.0,
                    ):
                        logger.error(
                            f"Error in git fetch! Sticking with {repo_commit_hash=} instead of {commit_hash=}"
                        )
                        keep_going = False

                    if keep_going and not cls._run_command_with_check(
                        ["git", "reset", "--hard"], target_directory
                    ):
                        logger.error(
                            f"Error in git reset! Sticking with {repo_commit_hash=} instead of {commit_hash=}"
                        )
                        keep_going = False

                    if keep_going:
                        if cls._run_command_with_check(
                            ["git", "checkout", commit_hash], target_directory
                        ):
                            repo_commit_hash = commit_hash
                        else:
                            logger.error(
                                f"Error in git checkout! Sticking with {repo_commit_hash=} instead of {commit_hash=}"
                            )

            # pull the lfs files
            cls._pull_lfs_files(target_directory)

            # get all the files in the repo
            files = cls._list_files(target_directory)
            files_with_3d_extension = [
                file
                for file in files
                if any(file.lower().endswith(ext) for ext in FILE_EXTENSIONS)
            ]

            # get the sha256 for each file
            file_hashes = []
            for file in tqdm(files_with_3d_extension, desc="Handling 3D object files"):
                file_hash = get_file_hash(file)
                # remove the temp_dir from the file path
                github_url = file.replace(
                    target_directory,
                    f"https://github.com/{org}/{repo}/blob/{repo_commit_hash}",
                )
                file_hashes.append(dict(sha256=file_hash, fileIdentifier=github_url))

                # handle the object under different conditions
                if github_url in expected_objects:
                    out[github_url] = file[len(target_directory) + 1 :]
                    if expected_objects[github_url] == file_hash:
                        if handle_found_object is not None:
                            handle_found_object(
                                local_path=file,
                                file_identifier=github_url,
                                sha256=file_hash,
                                metadata=dict(
                                    github_organization=org, github_repo=repo
                                ),
                            )
                    else:
                        if handle_modified_object is not None:
                            handle_modified_object(
                                local_path=file,
                                file_identifier=github_url,
                                new_sha256=file_hash,
                                old_sha256=expected_objects[github_url],
                                metadata=dict(
                                    github_organization=org, github_repo=repo
                                ),
                            )
                elif handle_new_object is not None:
                    handle_new_object(
                        local_path=file,
                        file_identifier=github_url,
                        sha256=file_hash,
                        metadata=dict(github_organization=org, github_repo=repo),
                    )

            # save the file hashes to a json file
            with open(
                os.path.join(target_directory, ".objaverse-file-hashes.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(file_hashes, f, indent=2)

            # remove the .git directory
            shutil.rmtree(os.path.join(target_directory, ".git"))

            if save_repo_format is None:
                # remove the paths, since it's not downloaded
                out = {}
            else:
                logger.debug(f"Saving {org}/{repo} as {save_repo_format}")
                # save the repo to a zip file
                if save_repo_format == "zip":
                    shutil.make_archive(target_directory, "zip", target_directory)
                elif save_repo_format == "tar":
                    with tarfile.open(
                        os.path.join(temp_dir, f"{repo}.tar"), "w"
                    ) as tar:
                        tar.add(target_directory, arcname=repo)
                elif save_repo_format == "tar.gz":
                    with tarfile.open(
                        os.path.join(temp_dir, f"{repo}.tar.gz"), "w:gz"
                    ) as tar:
                        tar.add(target_directory, arcname=repo)
                elif save_repo_format == "files":
                    pass
                else:
                    raise ValueError(
                        f"save_repo_format must be one of zip, tar, tar.gz, files. Got {save_repo_format}"
                    )

                dirname = os.path.join(base_dir, "repos", org)
                fs.makedirs(dirname, exist_ok=True)
                if save_repo_format != "files":
                    # move the repo to the correct location (with put)
                    fs.put(
                        os.path.join(temp_dir, f"{repo}.{save_repo_format}"),
                        os.path.join(dirname, f"{repo}.{save_repo_format}"),
                    )

                    for file_identifier in out.copy():
                        out[file_identifier] = os.path.join(
                            dirname, f"{repo}.{save_repo_format}", out[file_identifier]
                        )
                else:
                    # move the repo to the correct location (with put)
                    fs.put(target_directory, dirname, recursive=True)

                    for file_identifier in out.copy():
                        out[file_identifier] = os.path.join(
                            dirname, repo, out[file_identifier]
                        )

        # get each object that was missing from the expected objects
        if handle_missing_object is not None:
            obtained_urls = {x["fileIdentifier"] for x in file_hashes}
            for github_url, sha256 in expected_objects.items():
                if github_url not in obtained_urls:
                    handle_missing_object(
                        file_identifier=github_url,
                        sha256=sha256,
                        metadata=dict(github_organization=org, github_repo=repo),
                    )

        return out

    @classmethod
    def _list_files(cls, root_dir: str) -> List[str]:
        files = []
        for root, dirs, filenames in os.walk(root_dir, followlinks=False):
            for f in filenames:
                try:
                    file_path = os.path.join(root, f)
                    # Check if the file exists and is not a broken symlink
                    if os.path.exists(file_path) or not os.path.islink(file_path):
                        files.append(file_path)
                except (OSError, FileNotFoundError):
                    # Skip files that can't be accessed (broken symlinks, permission issues, etc.)
                    logger.debug(f"Skipping inaccessible file: {os.path.join(root, f)}")
                    continue
        return files

    @classmethod
    def _pull_lfs_files(cls, repo_dir: str) -> None:
        if cls._has_lfs_files(repo_dir):
            cls._run_command_with_check(
                ["git", "lfs", "pull"],
                cwd=repo_dir,
                command_name="git lfs pull",
                max_retries=5,
                retry_delay=2.0,
            )

    @classmethod
    def _has_lfs_files(cls, repo_dir: str) -> bool:
        gitattributes_path = os.path.join(repo_dir, ".gitattributes")
        if not os.path.exists(gitattributes_path):
            return False
        with open(gitattributes_path, "r", encoding="utf-8") as f:
            for line in f:
                if "filter=lfs" in line:
                    return True
        return False

    @classmethod
    def _get_commit_hash_from_local_git_dir(cls, local_git_dir: str) -> str:
        """Get the commit hash of the local git directory."""
        # get the git hash of the repo
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=local_git_dir,
            capture_output=True,
            check=True,
        )
        commit_hash = result.stdout.strip().decode("utf-8")
        return commit_hash

    @classmethod
    def _parallel_process_repo(cls, args) -> Dict[str, str]:
        """Helper function to process a repo in parallel.

        Note: This function is used to parallelize the processing of repos. It is not
        intended to be called directly.

        Args:
            args (Tuple): Tuple of arguments to pass to _process_repo.

        Returns:
            Dict[str, str]: A dictionary that maps from the "fileIdentifier" to the path
                of the downloaded object.
        """

        (
            repo_id_hash,
            fs,
            base_dir,
            save_repo_format,
            expected_objects,
            handle_found_object,
            handle_modified_object,
            handle_missing_object,
            handle_new_object,
        ) = args
        repo_id = "/".join(repo_id_hash.split("/")[:2])
        commit_hash = repo_id_hash.split("/")[2]
        return cls._process_repo(
            repo_id=repo_id,
            fs=fs,
            base_dir=base_dir,
            save_repo_format=save_repo_format,
            expected_objects=expected_objects,
            handle_found_object=handle_found_object,
            handle_modified_object=handle_modified_object,
            handle_missing_object=handle_missing_object,
            handle_new_object=handle_new_object,
            commit_hash=commit_hash,
        )

    @classmethod
    def _process_group(cls, group):
        key, group_df = group
        return key, group_df.set_index("fileIdentifier")["sha256"].to_dict()

    @classmethod
    def download_objects(
        cls,
        objects: pd.DataFrame,
        download_dir: Optional[str] = "~/.objaverse",
        processes: Optional[int] = None,
        handle_found_object: Optional[Callable] = None,
        handle_modified_object: Optional[Callable] = None,
        handle_missing_object: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Download the specified GitHub objects.

        Args:
            objects (pd.DataFrame): GitHub objects to download. Must have columns for
                the object "fileIdentifier" and "sha256". Use the `get_annotations`
                function to get the metadata.
            download_dir (Optional[str], optional): Directory to download the GitHub
                objects to. Supports all file systems supported by fsspec. If None, the
                repository will not be saved (note that save_repo_format must also be
                None in this case, otherwise a ValueError is raised). Defaults to
                "~/.objaverse".
            processes (Optional[int], optional): Number of processes to use for
                downloading.  If None, will use the number of CPUs on the machine.
                Defaults to None.
            handle_found_object (Optional[Callable], optional): Called when an object is
                successfully found and downloaded. Here, the object has the same sha256
                as the one that was downloaded with Objaverse-XL. If None, the object
                will be downloaded, but nothing will be done with it. Args for the
                function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): GitHub URL of the 3D object.
                - sha256 (str): SHA256 of the contents of the 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used. Defaults to None.
            handle_modified_object (Optional[Callable], optional): Called when a
                modified object is found and downloaded. Here, the object is
                successfully downloaded, but it has a different sha256 than the one that
                was downloaded with Objaverse-XL. This is not expected to happen very
                often, because the same commit hash is used for each repo. If None, the
                object will be downloaded, but nothing will be done with it. Args for
                the function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): GitHub URL of the 3D object.
                - new_sha256 (str): SHA256 of the contents of the newly downloaded 3D
                    object.
                - old_sha256 (str): Expected SHA256 of the contents of the 3D object as
                    it was when it was downloaded with Objaverse-XL.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used. Defaults to None.
            handle_missing_object (Optional[Callable], optional): Called when an object
                that is in Objaverse-XL is not found. Here, it is likely that the
                repository was deleted or renamed. If None, nothing will be done with
                the missing object. Args for the function include:
                - file_identifier (str): GitHub URL of the 3D object.
                - sha256 (str): SHA256 of the contents of the original 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used. Defaults to None.
            save_repo_format (Optional[Literal["zip", "tar", "tar.gz", "files"]],
                optional): Format to save the repository. If None, the repository will
                not be saved. If "files" is specified, each file will be saved
                individually. Otherwise, the repository can be saved as a "zip", "tar",
                or "tar.gz" file. Defaults to None.
            handle_new_object (Optional[Callable], optional): Called when a new object
                is found. Here, the object is not used in Objaverse-XL, but is still
                downloaded with the repository. The object may have not been used
                because it does not successfully import into Blender. If None, the
                object will be downloaded, but nothing will be done with it. Args for
                the function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): GitHub URL of the 3D object.
                - sha256 (str): SHA256 of the contents of the 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used. Defaults to None.
            skip_existing_on_s3 (bool, optional): If True, skip downloading repositories
                when any mapped S3 object already exists for the repo's file identifiers.
                Defaults to False.
            s3_mapping_path (str, optional): Path to the pickle file mapping file
                identifiers to S3 prefixes. Only used when skip_existing_on_s3 is True.
                Defaults to "/home/ray/mappings.pkl".

        Raises:
            ValueError: If download_dir is None and save_repo_format is not None.
                Otherwise, we don't know where to save the repo!

        Returns:
            Dict[str, str]: A dictionary that maps from the "fileIdentifier" to the path
                of the downloaded object.
        """
        save_repo_format = kwargs.get("save_repo_format", None)
        handle_new_object = kwargs.get("handle_new_object", None)
        skip_existing_on_s3 = kwargs.get("skip_existing_on_s3", False)
        s3_mapping_path = kwargs.get("s3_mapping_path", "/home/ray/mappings.pkl")

        if processes is None:
            processes = multiprocessing.cpu_count()
        if download_dir is None:
            if save_repo_format is not None:
                raise ValueError(
                    f"If {save_repo_format=} is not None, {download_dir=} must be specified."
                )
            # path doesn't matter if we're not saving the repo
            download_dir = "~/.objaverse"

        base_download_dir = os.path.join(download_dir, "github")
        fs, path = fsspec.core.url_to_fs(base_download_dir)
        fs.makedirs(path, exist_ok=True)

        # Getting immediate subdirectories of root_path
        if save_repo_format == "files":
            downloaded_repo_dirs = fs.glob(base_download_dir + "/repos/*/*/")
            downloaded_repo_ids = {
                "/".join(x.split("/")[-2:]) for x in downloaded_repo_dirs
            }
        else:
            downloaded_repo_dirs = fs.glob(
                base_download_dir + f"/repos/*/*.{save_repo_format}"
            )
            downloaded_repo_ids = set()
            for x in downloaded_repo_dirs:
                org, repo = x.split("/")[-2:]
                repo = repo[: -len(f".{save_repo_format}")]
                repo_id = f"{org}/{repo}"
                downloaded_repo_ids.add(repo_id)

        # make copy of objects
        objects = objects.copy()

        # get the unique repoIds
        objects["repoIdHash"] = objects.apply(cls._get_repo_id_with_hash, axis=1)
        repo_id_hashes = set(objects["repoIdHash"].unique().tolist())
        repo_ids = {
            "/".join(repo_id_hash.split("/")[:2]) for repo_id_hash in repo_id_hashes
        }
        assert len(repo_id_hashes) == len(repo_ids), (
            f"More than 1 commit hash per repoId!"
            f" {len(repo_id_hashes)=}, {len(repo_ids)=}"
        )

        logger.info(
            f"Provided {len(repo_ids)} repoIds with {len(objects)} objects to process."
        )

        # remove repoIds that have already been downloaded
        repo_ids_to_download = repo_ids - downloaded_repo_ids
        repo_id_hashes_to_download = [
            repo_id_hash
            for repo_id_hash in repo_id_hashes
            if "/".join(repo_id_hash.split("/")[:2]) in repo_ids_to_download
        ]

        logger.info(
            f"Found {len(repo_ids_to_download)} repoIds not yet downloaded. Downloading now..."
        )

        # get the objects to download
        groups = list(objects.groupby("repoIdHash"))
        with Pool(processes=processes) as pool:
            out_list = list(
                tqdm(
                    pool.imap_unordered(cls._process_group, groups),
                    total=len(groups),
                    desc="Grouping objects by repository",
                )
            )
        objects_per_repo_id_hash = dict(out_list)

        if skip_existing_on_s3 and repo_id_hashes_to_download:
            uuid_mappings = cls._load_uuid_mappings(s3_mapping_path)
            path_exists_cache: Dict[str, bool] = {}
            filtered_repo_id_hashes = []
            skipped_count = 0

            logger.info(f"Checking {len(repo_id_hashes_to_download)} repoIds for existing objects on S3.")
            for repo_id_hash in tqdm(repo_id_hashes_to_download, desc="Checking repoIds for existing objects on S3"):
                repo_objects = objects_per_repo_id_hash.get(repo_id_hash, {})
                if cls._repo_has_existing_objects_on_s3(
                    repo_objects, uuid_mappings, path_exists_cache
                ):
                    skipped_count += 1
                    continue
                filtered_repo_id_hashes.append(repo_id_hash)

            if skipped_count:
                logger.info(
                    "Skipping {} repoIds because objects already exist on S3.",
                    skipped_count,
                )

            repo_id_hashes_to_download = filtered_repo_id_hashes

            if not repo_id_hashes_to_download:
                logger.info("No repositories left to download after S3 filtering.")
                return {}

        all_args = [
            (
                repo_id_hash,
                fs,
                path,
                save_repo_format,
                objects_per_repo_id_hash[repo_id_hash],
                handle_found_object,
                handle_modified_object,
                handle_missing_object,
                handle_new_object,
            )
            for repo_id_hash in repo_id_hashes_to_download
        ]

        with Pool(processes=processes) as pool:
            # use tqdm to show progress
            out = list(
                tqdm(
                    pool.imap_unordered(cls._parallel_process_repo, all_args),
                    total=len(all_args),
                    desc="Downloading repositories",
                )
            )

        out_dict = {}
        for x in out:
            out_dict.update(x)

        return out_dict
