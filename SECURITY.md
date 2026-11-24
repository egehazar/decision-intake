# Security

DecisionIntake's hosted service (https://decision-intake.vercel.app) stores nothing: no accounts, no
database, no answers or states in logs. The state string a client sends back is not secret or signed;
it holds what the person said, so integrations should treat it like the answers themselves.

Framing: every page refuses to be framed (`frame-ancestors 'none'`) except `/embed`, the screener made for
other sites (DECISIONS D47), which HTTPS pages may frame. It has no account, no stored data and no action
beyond answering questions, so a framing page can at worst show the screener under its own design. The
embedded page tells its host only its height. `embed.js` puts it in a sandboxed iframe with no referrer.

To report a vulnerability, please use the repository's **Security** tab (Report a vulnerability) rather
than a public issue, and include the request and the response you saw. If that option isn't shown, open
an issue that only asks for a private contact, without details. Reports about wrong
screening results are welcome as ordinary [issues](https://github.com/egehazar/decision-intake/issues).
