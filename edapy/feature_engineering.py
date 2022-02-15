import pandas as pd


def add_consecutive_days(df, col_ID, col_date, col_consecutive='consecutive'):
    """
    Add number of consecutive transaction column to dataframe of transaction.

    Arguments:
        df {pd.DataFrame} -- typically transaction dataframe
        col_ID {String} -- column of unique user ID
        col_date {String} -- column of timestamp or datetime identifying when the transaction occurs

    Keyword Arguments:
        col_consecutive {String} -- name of result column
    """

    # Convert date column to pd.DateTime format
    df[col_date] = pd.to_datetime(df[col_date])

    # Create date diff in days column
    df['date_diff'] = (max(df[col_date]) - df[col_date])
    df['date_diff'] = [x.days for x in df['date_diff']]

    # Initialize empty consecutive column
    df[col_consecutive] = [None] * df.shape[0]

    for ID in df[col_ID].unique():
        df_temp = df[df[col_ID] == ID]

        # Sort descending transactional data by date column, get unique date_diff column
        temp = sorted(df_temp['date_diff'].unique())[::-1]

        # For each value in temp, calulate the consecutive days
        res = [0]
        curr = 0
        for i in range(len(temp) - 1):
            if ((temp[i+1] - temp[i]) == -1):
                curr = curr + 1
            else:
                curr = 0
            res.append(curr)

        # Create dictionary for mapping it back to the data
        res2 = [[x1, x2] for x1, x2 in zip(temp, res)]
        res2 = dict(res2)

        # Finally, create the consecutive columns
        df.loc[df[col_ID] == ID, [col_consecutive]] = [res2[x] for x in df_temp['date_diff']]

    # Check if the result is correct
    df.sort_values(by='date_diff', ascending=False)
