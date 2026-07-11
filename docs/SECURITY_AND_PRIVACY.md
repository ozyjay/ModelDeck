# Security and privacy

All defaults bind to `127.0.0.1`. LAN exposure requires an explicit future decision and
threat review. Open Day mode forces downloads off.

The frontend cannot submit commands, executable paths, raw runtime arguments, environment
variables, tokens, arbitrary filesystem paths, Docker access, camera data, uploads, or
cloud endpoints. Worker IDs select prevalidated manifests. Subprocesses use argument
arrays without a shell.

Visitor prompts and generated content are not stored or logged. Supervisor log capture is
bounded to the latest 500 records per worker and redacts prompt, output, authorisation,
API-key, and token-shaped fields before persisting JSON Lines files under
`var/log/workers`. The location can be changed with `MODELDECK_LOG_DIR`. Full diagnostic
capture is not implemented in this slice. SQLite holds configuration and compatibility
evidence, not content history.
