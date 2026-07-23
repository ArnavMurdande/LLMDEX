# Power BI

Power BI should visualize processed LLMDEX outputs, not reimplement the ranking.

## Import

1. Publish or locate the raw URL for `data/index/latest.csv`.
2. In Power BI Desktop choose **Get data → Web**.
3. Enter the raw CSV URL.
4. Promote the first row as headers.
5. Set score/cost/speed columns to decimal number and rank/context columns to
   whole number as appropriate.
6. Keep blank cells as null.

For history, repeat with `data/history/family_snapshots.csv` and relate
`family_id` to the current index/family export.

Example Power Query:

```powerquery
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/ArnavMurdande/LLMDEX/main/data/index/latest.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    Headers
```

Recommended report pages: General ranking, consensus vs source percentile,
capabilities, price/speed, score history, source agreement, availability, and
data quality.

Do not replace null scores with zero and do not duplicate the consensus in DAX.
