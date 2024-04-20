from datetime import date, timedelta


def get_today_and_15_days_ago():
    # Get today's date
    today = date.today()

    # Calculate the date 15 days before today
    fifteen_days_ago = today - timedelta(days=15)

    # Print both dates in a user-friendly format (optional)
    print(f"Today's date: {today.strftime('%Y-%m-%d')}")
    print(f"Date 15 days ago: {fifteen_days_ago.strftime('%Y-%m-%d')}")

    return [today, fifteen_days_ago]