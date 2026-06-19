def is_engineered_column(col):
    suffixes = [
        "_scaled",
        "_robust",
        "_log",
        "_encoded",
        "_freq",
        "_year",
        "_month",
        "_day",
        "_weekday"
    ]

    return any(str(col).endswith(s) for s in suffixes)