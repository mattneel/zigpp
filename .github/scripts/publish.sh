#!/usr/bin/env bash
# Publishes the release of one Zig++ build.
#
#   .github/scripts/publish.sh <dist directory> <version> <commit> <owner/repo>
#
# The dist directory holds the build's archives, zig-<arch>-<os>-<version>.tar.xz or .zip.
# GH_TOKEN may write the repository's contents. RELEASE_TAG_KEY is the private key of the
# repository's release deploy key, which may push tags. Run it from a clone that has the
# commit.
#
# Builds are forever: a published release is never changed, and its tag,
# zigpp-<version without build metadata>, names that one build and never moves. So:
#
#   - a release that is published already is left as it is;
#   - a draft was never released: a run that failed before publishing left it, and its
#     assets are replaced with this run's before it is published;
#   - otherwise the tag is pushed at the commit, a draft is created on it, the assets are
#     uploaded to the draft, and the draft is published. The assets go up first because a
#     published release takes no more of them once releases are immutable.
#
# The tag is pushed with the deploy key rather than created by the release API with the
# workflow token: GitHub refuses the workflow token a ref at a commit whose workflow files
# differ from master's, which is every commit behind a master that has changed a workflow
# since.
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
: "${RELEASE_TAG_KEY:?is not set: the release deploy key pushes the tag}"

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

# The deploy key, and GitHub's own host keys, for every git command below.
ssh_dir=$(mktemp -d)
trap 'rm -rf "$ssh_dir"' EXIT
printf '%s\n' "$RELEASE_TAG_KEY" > "$ssh_dir/key"
chmod 600 "$ssh_dir/key"
gh api meta --jq '.ssh_keys[] | "github.com " + .' > "$ssh_dir/known_hosts"
export GIT_SSH_COMMAND="ssh -i $ssh_dir/key -o IdentitiesOnly=yes -o UserKnownHostsFile=$ssh_dir/known_hosts -o StrictHostKeyChecking=yes"
remote="git@github.com:$repo.git"

# The download index and the checksums the release carries.
"$here/index.py" "$dist" "$version" "$tag" "$repo"
(cd "$dist" && sha256sum zig-* index.json > SHA256SUMS)
assets=("$dist"/zig-* "$dist/index.json" "$dist/SHA256SUMS")

# The tag: pushed once, at this commit. A tag that names another commit is another build.
tagged=$(git ls-remote "$remote" "refs/tags/$tag" "refs/tags/$tag^{}" |
    awk -v ref="refs/tags/$tag" '
        $2 == ref "^{}" { peeled = $1 }
        $2 == ref { plain = $1 }
        END { print (peeled != "" ? peeled : plain) }')
if [ -z "$tagged" ]; then
    git push "$remote" "$commit:refs/tags/$tag"
elif [ "$tagged" != "$commit" ]; then
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
            id=$(gh api -X POST "repos/$repo/releases" -f tag_name="$tag" \
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
