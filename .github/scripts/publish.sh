#!/usr/bin/env bash
# Publishes the release of one Zig++ build.
#
#   .github/scripts/publish.sh <dist directory> <version> <commit> <owner/repo>
#
# The dist directory holds the build's archives, zig-<arch>-<os>-<version>.tar.xz or .zip.
# GH_TOKEN may write the repository's contents.
#
# Builds are forever: a published release is never changed, and its tag,
# zigpp-<version without build metadata>, names that one build and never moves. So:
#
#   - a release that is published already is left as it is;
#   - a draft was never released: a run that failed before publishing left it, and its
#     assets are replaced with this run's before it is published;
#   - otherwise a draft is created at the commit, the assets are uploaded to the draft, and
#     the draft is published, which creates the tag. The assets go up first because a
#     published release takes no more of them once releases are immutable.
#
# The latest release only moves forward. A release becomes the latest when its version is
# newer than the latest's, by the ordering of the download index, and at the end the newest
# published release is made the latest again, which settles two publishes that finished
# together.
set -euo pipefail

if [ $# -ne 4 ]; then
    echo "usage: $0 <dist directory> <version> <commit> <owner/repo>" >&2
    exit 2
fi
dist=$1 version=$2 commit=$3 repo=$4
here=$(cd "$(dirname "$0")" && pwd)
tag="zigpp-${version%%+*}"

# newest <tag>...: the tag of the newest version among the zigpp-* tags given, ordered as the
# download index orders its releases.
newest() {
    python3 - "$here" "$@" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from downloads import version_key
tags = [tag for tag in sys.argv[2:] if tag.startswith("zigpp-")]
if tags:
    print(max(tags, key=lambda tag: version_key({"version": tag[len("zigpp-"):]})))
EOF
}

# release_of <tag>: "<id> <draft>" of the release with that tag, drafts included, or nothing.
release_of() {
    gh api --paginate "repos/$repo/releases?per_page=100" \
        --jq ".[] | select(.tag_name == \"$1\") | \"\(.id) \(.draft)\""
}

# latest_tag: the tag of the repository's latest release, or nothing when it has none.
latest_tag() {
    local out
    if out=$(gh api "repos/$repo/releases/latest" --jq .tag_name 2>&1); then
        printf '%s\n' "$out"
    elif [[ $out == *"HTTP 404"* ]]; then
        return 0
    else
        printf '%s\n' "$out" >&2
        return 1
    fi
}

# The download index and the checksums the release carries.
"$here/index.py" "$dist" "$version" "$tag" "$repo"
(cd "$dist" && sha256sum zig-* index.json > SHA256SUMS)
assets=("$dist"/zig-* "$dist/index.json" "$dist/SHA256SUMS")

# A tag that names another commit is another build.
tagged=$(git ls-remote "https://github.com/$repo.git" "refs/tags/$tag" "refs/tags/$tag^{}" |
    awk -v ref="refs/tags/$tag" '
        $2 == ref "^{}" { peeled = $1 }
        $2 == ref { plain = $1 }
        END { print (peeled != "" ? peeled : plain) }')
if [ -n "$tagged" ] && [ "$tagged" != "$commit" ]; then
    echo "error: $tag names $tagged already, and this build is $commit" >&2
    exit 1
fi

read -r id draft < <(release_of "$tag") || true
case ${draft:-none} in
    false)
        echo "$tag is published already; a published build is never changed"
        ;;
    true | none)
        # The draft is addressed by its id from here on: a release just created is not
        # always in the list of releases yet.
        if [ "${draft:-none}" = true ]; then
            echo "$tag is a draft that was never published; its assets are replaced"
            for asset in $(gh api --paginate "repos/$repo/releases/$id/assets" --jq '.[].id'); do
                gh api -X DELETE "repos/$repo/releases/assets/$asset"
            done
        else
            id=$(gh api -X POST "repos/$repo/releases" -f tag_name="$tag" -f target_commitish="$commit" \
                -f name="Zig++ $version" -F draft=true -F generate_release_notes=true --jq .id)
        fi
        for file in "${assets[@]}"; do
            name=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${file##*/}")
            gh api -X POST "https://uploads.github.com/repos/$repo/releases/$id/assets?name=$name" \
                -H "Content-Type: application/octet-stream" --input "$file" --jq '"uploaded " + .name'
        done

        latest=$(latest_tag)
        make_latest=false
        if [ "$(newest "$tag" ${latest:+"$latest"})" = "$tag" ]; then make_latest=true; fi
        gh api -X PATCH "repos/$repo/releases/$id" -F draft=false -f make_latest=$make_latest >/dev/null
        echo "published $tag at $commit (latest: $make_latest)"
        ;;
esac

# The newest published release is the latest, whichever publish finished last.
published=$(gh api --paginate "repos/$repo/releases?per_page=100" \
    --jq '.[] | select(.draft == false and .prerelease == false) | .tag_name')
mapfile -t published <<< "$published"
best=$(newest "${published[@]}")
if [ -n "$best" ] && [ "$best" != "$(latest_tag)" ]; then
    read -r best_id _ < <(release_of "$best")
    gh api -X PATCH "repos/$repo/releases/$best_id" -f make_latest=true >/dev/null
    echo "made $best the latest release"
fi
