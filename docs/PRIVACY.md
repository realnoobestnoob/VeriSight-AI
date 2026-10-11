# Privacy notes
- Uploaded/captured images are decoded in memory and **not saved**.
- Stored per scan (only if `VERISIGHT_DATABASE_URL` is set): timestamp, input type, file type, optional SHA-256 of the file bytes (duplicate detection only, not identity), model release id, scores, decision status, latency, status/error category, whether an LLM was used. **No filenames, images or message text.**
- Message text is processed locally by rule-based checks. It is sent to an external LLM provider only if one is configured **and** the user ticks the opt-in box. The image is never sent.
- The public dashboard shows aggregates only; there is no global per-scan history. Session history lives in the visitor's browser session.
- Retention: define and publish a retention period for the `scans` table before a public launch (e.g. delete rows older than N days); this repo does not enforce one.
- Use a least-privilege database role (INSERT/SELECT on the two tables) for the app.
