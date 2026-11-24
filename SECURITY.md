# Security

DecisionIntake's hosted service (https://decision-intake.vercel.app) stores nothing: no accounts, no
database, no answers or states in logs. The state string a client sends back is not secret or signed;
it holds what the person said, so integrations should treat it like the answers themselves.

To report a vulnerability, please use the repository's **Security** tab (Report a vulnerability) rather
than a public issue, and include the request and the response you saw. If that option isn't shown, open
an issue that only asks for a private contact, without details. Reports about wrong
screening results are welcome as ordinary [issues](https://github.com/egehazar/decision-intake/issues).
