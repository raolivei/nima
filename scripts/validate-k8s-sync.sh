#!/bin/bash
# Validate that nima/k8s/ matches pi-fleet manifests
# This ensures emergency deployments will match GitOps state

PROJECT="nima"
PI_FLEET_PATH="../pi-fleet/clusters/eldertree/nima"

echo "🔍 Validating K8s Manifest Sync"
echo "================================"
echo "Comparing: $PROJECT/k8s/ ↔ pi-fleet/nima/"
echo ""

# Check if pi-fleet directory exists
if [ ! -d "$PI_FLEET_PATH" ]; then
    echo "❌ pi-fleet nima directory not found at: $PI_FLEET_PATH"
    exit 1
fi

# Files to exclude from comparison (only exist in one location)
EXCLUDE_ARGS=(
    --exclude="kustomization.yaml"      # Only in pi-fleet
    --exclude="ghcr-secret-external.yaml"  # Only in pi-fleet
    --exclude="secrets.yaml.example"    # Template only
)

# Compare directories
echo "Checking for differences..."
DIFF_OUTPUT=$(diff -r --brief k8s/ "$PI_FLEET_PATH/" "${EXCLUDE_ARGS[@]}" 2>&1 || true)

if [ -z "$DIFF_OUTPUT" ]; then
    echo "✅ Manifests are in sync!"
    echo ""
    echo "   nima/k8s/ matches pi-fleet/clusters/eldertree/nima/"
    echo "   Emergency deployment will match GitOps state."
    exit 0
else
    echo "⚠️  Manifests differ!"
    echo ""
    echo "$DIFF_OUTPUT"
    echo ""
    echo "❌ Files are out of sync."
    echo ""
    echo "💡 Action required:"
    echo "   1. Review the differences above"
    echo "   2. Sync the files manually"
    echo "   3. Commit changes to both locations"
    echo ""
    echo "   Example: cp k8s/deploy.yaml $PI_FLEET_PATH/"
    exit 1
fi



