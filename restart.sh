#!/bin/bash
# Stop the current local OntoCare process with Ctrl+C before running this script.
# Railway restarts are managed by deployment, not by this local helper.
set -e
cd "$(dirname "$0")"
exec bash run.sh
