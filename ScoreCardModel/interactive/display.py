def scorecard_to_html(card_df) -> str:
    """Format a scorecard DataFrame as an HTML table for Jupyter notebooks."""
    if card_df.empty:
        return "<p><i>Empty scorecard</i></p>"

    rows_html = ""
    for _, row in card_df.iterrows():
        woe_str = f"{row['WOE']:.4f}" if isinstance(row['WOE'], float) else str(row['WOE'])
        pts_str = f"{row['Points']:.2f}" if isinstance(row['Points'], float) else str(row['Points'])
        rows_html += (
            f"<tr style='border-bottom:1px solid #ddd'>"
            f"<td style='padding:4px 8px'>{row['Variable']}</td>"
            f"<td style='padding:4px 8px'>{row['Bin']}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{woe_str}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{pts_str}</td>"
            f"</tr>"
        )

    return (
        "<table style='border-collapse:collapse;width:100%;font-family:monospace;font-size:13px'>"
        "<thead><tr style='background:#f5f5f5;border-bottom:2px solid #ccc'>"
        "<th style='padding:6px 8px;text-align:left'>Variable</th>"
        "<th style='padding:6px 8px;text-align:left'>Bin</th>"
        "<th style='padding:6px 8px;text-align:right'>WOE</th>"
        "<th style='padding:6px 8px;text-align:right'>Points</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


def _card_to_html(self) -> str:
    """_repr_html_ hook for ScoreCardTransformer."""
    try:
        card = self.export_scorecard()
        return scorecard_to_html(card)
    except Exception:
        return None
