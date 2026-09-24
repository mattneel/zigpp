#!/usr/bin/env python3
"""Writes the cumulative download index of Zig++ and the book's Downloads page.

    .github/scripts/downloads.py --releases <dir> --page <file> --index <file>

Every release publishes an index.json in the format of upstream's
https://ziglang.org/download/index.json: one "master" entry that names the
version, its date, and the tarball, SHA-256, and size of every target. The files
in the release directory are those indexes, one per release. This script merges
them into one cumulative index, newest first, and writes the Downloads chapter
of the book from the same entries: the newest release, one section per release
with a table of its archives and their hashes, and how to verify a download.

mdbook renders the page, which is why it is plain CommonMark, and why no table
holds a hash: a 64-character cell is far too wide for the page.
"""

import argparse
import json
import os
import sys
import urllib.parse

# The targets of every release, in the order the page lists them. Any other
# target comes after these, alphabetically.
TARGETS = ("x86_64-linux", "aarch64-linux", "aarch64-macos", "x86_64-windows")

# The owner/repo of the release pages, for prose that has no tarball URL to read.
DEFAULT_REPO = "mattneel/zigpp"

# The "what a release contains" block, with and without the Installing chapter.
DETAILS_WITH_CHAPTER = [
    "An archive is named `zig-<arch>-<os>-<version>.tar.xz`, or `.zip` on Windows. It holds the",
    "`zig` executable, `lib/`, `doc/langref.html`, `LICENSE`, and `README.md`. The",
    "[Installing](installing.html) chapter covers how the compiler finds the `lib/` it carries.",
]

DETAILS_WITHOUT_CHAPTER = [
    "An archive is named `zig-<arch>-<os>-<version>.tar.xz`, or `.zip` on Windows. It holds the",
    "`zig` executable, `lib/`, `doc/langref.html`, `LICENSE`, and `README.md`. The compiler finds",
    "the `lib/` beside the `zig` executable, so unpack the archive as a whole.",
]


def parse_args():
    """The command line: where the release indexes are, and what to write."""
    parser = argparse.ArgumentParser(
        description="Merges the release indexes of Zig++ into the Downloads page and index.",
    )
    parser.add_argument(
        "--releases",
        required=True,
        metavar="DIR",
        help="directory of per-release index.json files, one per release",
    )
    parser.add_argument(
        "--page",
        required=True,
        metavar="FILE",
        help="markdown file to write the Downloads chapter to",
    )
    parser.add_argument(
        "--index",
        required=True,
        metavar="FILE",
        help="JSON file to write the cumulative index to",
    )
    parser.add_argument(
        "--repo",
        default=None,
        metavar="OWNER/REPO",
        help="owner/repo of the releases, when no archive URL names it",
    )
    return parser.parse_args()


def read_release(path):
    """The "master" entry of the release index in a file, or a fatal error.

    A file that does not hold an index is an error, not a release to skip: the
    workflow downloads these files from the releases, so one that does not
    parse means a broken download, and the page would silently lose a release.
    """
    try:
        with open(path, encoding="utf-8") as file:
            index = json.load(file)
    except OSError as error:
        sys.exit(f"{path}: cannot read: {error.strerror or error}")
    except json.JSONDecodeError as error:
        sys.exit(f"{path}: invalid JSON: {error}")
    if not isinstance(index, dict) or not isinstance(index.get("master"), dict):
        sys.exit(f'{path}: no "master" entry; not a release index')
    return index["master"]


def read_releases(directory):
    """The entries of every index.json in a directory, in a stable order."""
    if not os.path.isdir(directory):
        return []
    entries = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name.endswith(".json") and os.path.isfile(path):
            entries.append(read_release(path))
    return entries


def version_key(entry):
    """A sort key that puts the newest version first.

    Build metadata is ignored and a pre-release is older than the release
    itself, as semver says: identifiers of a pre-release compare numerically
    when both are numeric, a numeric identifier is older than an alphanumeric
    one, and a shorter set of otherwise equal identifiers is older.
    """
    version = entry.get("version", "")
    core, _, pre = version.partition("+")[0].partition("-")
    numbers = tuple(int(part) for part in core.split(".") if part)
    if not pre:
        return (numbers, 1, (), version)
    identifiers = tuple(
        (0, int(identifier), "") if identifier.isdigit() else (1, 0, identifier)
        for identifier in pre.split(".")
    )
    return (numbers, 0, identifiers, version)


def newest_first(releases):
    """The releases, newest first, with the version as the final tie-breaker."""
    return sorted(releases, key=version_key, reverse=True)


def targets(entry):
    """The target names of a release: the usual four, then any others."""
    names = [key for key, value in entry.items() if isinstance(value, dict)]
    others = sorted(name for name in names if name not in TARGETS)
    return [name for name in TARGETS if name in names] + others


def tarball_parts(release):
    """The owner/repo and tag that a release's first tarball URL names."""
    for target in targets(release):
        tarball = release[target].get("tarball")
        if not tarball:
            continue
        parts = urllib.parse.urlparse(tarball).path.strip("/").split("/")
        if len(parts) == 6 and parts[2:4] == ["releases", "download"]:
            return f"{parts[0]}/{parts[1]}", urllib.parse.unquote(parts[4])
    return None


def release_source(release, repo):
    """The owner/repo and tag of a release, whose page the section links to.

    The tarball URLs name both. A release without archives has no URL to read,
    so its tag follows the release workflow's rule of naming the base version,
    and the repo is the one the caller gave.
    """
    parts = tarball_parts(release)
    if parts:
        return parts
    return repo, "zigpp-" + release.get("version", "").partition("+")[0]


def file_name(tarball):
    """The archive file name in a tarball URL, percent-decoded."""
    return urllib.parse.unquote(tarball.rsplit("/", 1)[-1]) if tarball else ""


def format_size(size):
    """A size in bytes as MiB with one decimal, as the table shows it."""
    try:
        count = int(size)
    except (TypeError, ValueError):
        return "" if size is None else str(size)
    return f"{count / (1 << 20):.1f} MiB"


def cell(text):
    """Text as a table cell: an unescaped pipe would end the cell."""
    return text.replace("|", "\\|")


def has_chapter(page, name):
    """Whether the book has the chapter named by a file beside the page."""
    directory = os.path.dirname(page)
    return os.path.exists(os.path.join(directory, name) if directory else name)


def details_block(installing):
    """The block that says what a release contains, and what SHA256SUMS covers."""
    body = DETAILS_WITH_CHAPTER if installing else DETAILS_WITHOUT_CHAPTER
    return [
        "<details>",
        "<summary>What a release contains</summary>",
        "",
        *body,
        "",
        "`SHA256SUMS` lists the SHA-256 of every archive and of `index.json`.",
        "</details>",
    ]


def release_section(release, repo):
    """One release's section: its date, its page, its archives, and their hashes."""
    owner, tag = release_source(release, repo)
    lines = [
        f"## {release.get('version', '')}",
        "",
        f"{release.get('date', '')} · "
        f"[{tag}](https://github.com/{owner}/releases/tag/{urllib.parse.quote(tag)})",
        "",
    ]
    names = targets(release)
    if not names:
        return lines
    lines += ["|Target|Archive|Size|", "|---|---|---|"]
    for target in names:
        asset = release[target]
        tarball = asset.get("tarball", "")
        lines.append(
            f"|{cell(target)}|[{cell('`' + file_name(tarball) + '`')}]({tarball})"
            f"|{cell(format_size(asset.get('size')))}|"
        )
    lines += ["", "`SHA256SUMS`:", "", "```"]
    for target in sorted(names, key=lambda name: file_name(release[name].get("tarball", ""))):
        asset = release[target]
        lines.append(f"{asset.get('shasum', '')}  {file_name(asset.get('tarball', ''))}")
    lines += ["```", ""]
    return lines


def render_page(releases, repo, installing):
    """The Downloads chapter: the newest release, then every release's section."""
    lines = ["# Downloads", ""]
    index_link = (
        "The machine-readable index of every release is at "
        "[download/index.json](download/index.json)."
    )
    if releases:
        newest = releases[0]
        lines += [
            "Zig++ publishes a release on every push to master, for x86_64-linux, aarch64-linux,",
            "aarch64-macos, and x86_64-windows. The newest is "
            f"{newest.get('version', '')}, published {newest.get('date', '')}.",
            "",
            *details_block(installing),
            "",
        ]
        for release in releases:
            lines += release_section(release, repo)
        lines += [
            "## Verifying a download",
            "",
            "```sh",
            "sha256sum -c SHA256SUMS",
            "```",
            "",
            "The archives and `SHA256SUMS` of a release are its assets, so download them into one",
            "directory first. The archives are a Zig++ compiler and its `lib/`: unpack one anywhere",
            "and run `zig` from it, with no installation step"
            + (" (see the [Installing](installing.html) chapter)." if installing else "."),
            "",
            "The machine-readable index of every release is at",
            "[download/index.json](download/index.json):",
            "`master` is the newest release, and every release has a key of its own.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        "Zig++ has no published release yet. The",
        f"[Release workflow](https://github.com/{repo}/blob/master"
        "/.github/workflows/release.yml)",
        "publishes one on every push to master, for x86_64-linux, aarch64-linux, aarch64-macos,",
        "and x86_64-windows; the newest one appears here. The standard library documentation",
        "and the language reference on this site come from that release, so they appear with it.",
        "",
        *details_block(installing),
        "",
        index_link,
        "",
    ]
    return "\n".join(lines)


def write_text(path, text):
    """Writes a file, with the directories it needs."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    args = parse_args()
    releases = newest_first(read_releases(args.releases))
    repo = args.repo or DEFAULT_REPO
    index = {}
    if releases:
        index["master"] = releases[0]
        for release in releases:
            index[release.get("version", "")] = release
    write_text(args.index, json.dumps(index, indent=2) + "\n")
    write_text(args.page, render_page(releases, repo, has_chapter(args.page, "installing.md")))


if __name__ == "__main__":
    main()
