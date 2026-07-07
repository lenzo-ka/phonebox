#!/usr/bin/env bash
#
# Bundle phonebox for deployment.
#
# Creates a standalone Python G2P predictor with the model embedded.
# The bundled file has zero dependencies and can run anywhere a Python
# interpreter is available.

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
phonebox_root="${script_dir}/.."

model='models/en_US_pocketsphinx.g2p.gz'
output=''

usage() {
    cat << 'EOF'
Usage: bundle_for_deployment.sh [OPTIONS] -o OUTPUT

Bundle phonebox model into a standalone Python file for deployment.

Options:
    -o, --output FILE       Output .py file (required)
    -m, --model FILE        Model file to bundle (default: models/en_US_pocketsphinx.g2p.gz)
    -h, --help              Show this help

Examples:
    bundle_for_deployment.sh -o g2p.py
    bundle_for_deployment.sh -m models/fr_FR.g2p.gz -o fr_g2p.py
EOF
}

while [[ ${#} -gt 0 ]]; do
    case "${1}" in
        -o|--output)
            output="${2}"
            shift 2
            ;;
        -m|--model)
            model="${2}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: ${1}"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${output}" ]]; then
    echo 'Error: -o/--output is required'
    usage
    exit 1
fi

if [[ ! -f "${phonebox_root}/${model}" ]]; then
    echo "Error: Model not found: ${model}"
    exit 1
fi

cd "${phonebox_root}"
phonebox bundle "${model}" -o "${output}"

echo ''
echo "Done: ${output}"
ls -lh "${output}"
