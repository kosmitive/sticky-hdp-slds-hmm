# Close snapshots

Both files are daily closes from the Yahoo Finance chart API (`interval=1d`), downloaded 2026-10-11. The forecast scripts read these files and do not download anything.

- `sp500_daily.csv`: S&P 500 index `^GSPC`, 1990-01-02 through 2026-10-09 (9261 rows). The only gap longer than five days is the market closure from 2001-09-10 to 2001-09-17. That closure is real and is left in the file.
- `btc_daily.csv`: `BTC-USD`, 2014-09-17 through 2026-10-10 (4407 rows). The download also contained 2026-10-11, the download date. That row is dropped because the session was still incomplete. Every remaining date is the next calendar day.

The fit uses every observation before the holdout: the last 250 trading days for the S&P 500, and the last 365 calendar days for Bitcoin. The training window is the rest of each file, which is many years, and it was not shortened.
