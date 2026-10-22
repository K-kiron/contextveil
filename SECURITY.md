# Security

Do not attach real personal data, passphrases or recovery vaults to public bug
reports. Reproduce defects with synthetic fixtures. Use a maintainer-configured
private reporting channel for vulnerabilities when one is available.

This initial version has local automated tests and adversarial code review; it
has not undergone an independent penetration test or production deployment audit.
See [the assurance contract](docs/assurance.md) before processing sensitive data.

Runtime inference is local. Installing optional dependencies/models accesses
their distribution servers. Custom detectors and task callbacks run with the
caller's permissions and can access the network; audit them before use.

Treat input documents as untrusted. Resource bounds reduce accidental overload
but do not sandbox document parsers. A service accepting hostile files should
place parsing and inference in an isolated worker with enforced time and memory
limits. Protect source files, policy/glossary files, output reports and recovery
material according to the data they contain.
