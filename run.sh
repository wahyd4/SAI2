#! /bin/bash
set -euo pipefail

sanic server:app --reload --debug --port 8800
