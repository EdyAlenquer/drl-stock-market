import datetime as dt
# from datetime import date

def add_years(date, years):
    try:
    #Return same day of the current year        
        return date.replace(year = date.year + years)
    except ValueError:
    #If not same day, it will return other, i.e.  February 29 to March 1 etc.        
        return date + (date(date.year + years, 1, 1) - date(date.year, 1, 1))

def add_years_from_isoformat(date, years):
    date = dt.datetime.fromisoformat(date)    
    
    try:
        #Return same day of the current year        
        date = date.replace(year = date.year + years)
    except ValueError:
        #If not same day, it will return other, i.e.  February 29 to March 1 etc.        
        date = date + (date(date.year + years, 1, 1) - date(date.year, 1, 1))
    
    return date.strftime('%Y-%m-%d')
    