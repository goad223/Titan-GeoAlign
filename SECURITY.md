# Security Policy

## Supported Versions

We release security patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability in Titan-GeoAlign, please report it responsibly:

1. **GitHub Private Vulnerability Reporting** (preferred):
   Navigate to [Security Advisories](https://github.com/goad223/Titan-GeoAlign/security/advisories/new)
   and submit a new advisory.

2. **Email**: If you are unable to use GitHub's advisory system, send an encrypted
   email to the maintainers describing the issue.

### What to include

Please provide as much of the following information as possible to help us triage
and resolve the issue quickly:

- Type of vulnerability (e.g., remote code execution, path traversal, deserialization)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact assessment — what an attacker could achieve

### Response timeline

| Milestone | Target time |
|-----------|-------------|
| Acknowledgement | 48 hours |
| Initial assessment | 5 business days |
| Fix or mitigation | 30 days (critical), 90 days (others) |
| Public disclosure | After fix is released |

We follow a **coordinated disclosure** policy. We ask that you give us adequate time
to remediate before publishing details of any vulnerability.

## Security Best Practices for Users

- Always pin dependency versions in production deployments.
- Do not expose the FastAPI server directly to the internet without authentication middleware.
- Store credentials and API keys in environment variables or a secrets manager — never in source code.
- Review the [OWASP Top 10](https://owasp.org/www-project-top-ten/) when integrating Titan-GeoAlign into web-facing services.
- Keep the `GDAL` and `rasterio` system libraries up-to-date; geospatial parsing libraries can be a vector for malformed-file attacks.

## Scope

The following are **in scope** for security reports:

- The `titan_geoalign` Python package and its CLI (`titan-align`)
- The FastAPI REST API (`src/titan_geoalign/api/`)
- The gRPC service definitions
- Docker images published under the project

The following are **out of scope**:

- Third-party dependencies (report those to the respective upstream projects)
- Issues in sample / example data files
- Denial-of-service attacks that require an authenticated user with write access

## Attribution

This security policy is adapted from the
[GitHub Security Policy template](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository).
