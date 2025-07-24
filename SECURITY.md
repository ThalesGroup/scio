# Good practices to follow

Since our framework heavily relies on [PyTorch](https://pytorch.org/), we invite users to be aware about [using PyTorch securely](https://github.com/pytorch/pytorch/security#using-pytorch-securely).

⚠️ **You must never store credentials information into source code or config file in a GitHub repository**
- Block sensitive data being pushed to GitHub by git-secrets or its likes as a git pre-commit hook
- Audit for slipped secrets with dedicated tools
- Use environment variables for secrets in CI/CD (e.g. GitHub Secrets) and secret managers in production

# Security Policy

## Supported Versions

As of today, only the latest version is being supported with security updates.

## Reporting and disclosing a Vulnerability

If you believe you have found a security vulnerability in `scio`, we encourage you to report it in a dedicated [security issue](https://github.com/ThalesGroup/scio/issues/new?labels=security) or to contact [security@opensource.thalesgroup.com](mailto:security@opensource.thalesgroup.com) right away. It is **important** that every publicly shared concern remain high-level as to avoid spreading exploitation steps.

We will investigate all legitimate reports and do our best to quickly fix the problem.

## Security Update policy

If a security vulnerability is found, users can expect a related tracking [announcement](https://github.com/ThalesGroup/scio/discussions/categories/announcements). Further releases will also mitigate the problem, possibly by removing related features until a proper fix is implemented.

## Known security gaps & future enhancements

None
