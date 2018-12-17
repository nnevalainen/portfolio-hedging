
import numpy as np
import pandas as pd

def clean_sheet(sheet):
    """
    Normalize data and destructure features to respective variables
    
    params:
        sheet (pd.DataFrame):   Excel worksheet
    
    returns:
        sheet (pd.DataFrame):   Modified sheet
        variables:
            mat (pd.Series):    (n,1) Days to maturity
            T (pd.Series):      (n,1) Years to maturity
            S (pd.Series):      (n,1) Stock prices
            Cobs (np.ndarray):  (n,m) Call prices
            E (pd.Series):      (1,m) Strike prices
            r (pd.Series):      (n,1) Risk-free interest rates
    """
    
    # normalize strikes
    strikes = np.array(sheet.columns[1:-3])
    nstrikes = strikes / 1000
    
    sheet['time'] = sheet.mat / 252  # years to maturity
    sheet.s_price /= 1000            # normalize stock prices
    sheet[strikes] /= 1000           # normalize call prices
    sheet.r /= 100                   # risk-free interest rate to percentage
    
    # convert to datetime
    sheet.date = pd.to_datetime(sheet.date)
    
    def replace_outlier(column):
        column[column > 0] /= 1000
        
    # remove call outliers
    sheet[strikes].apply(lambda col: col[col > 1] /1000)
    
    # rename strike columns with normalized strikes
    sheet = sheet.rename(columns={key: key/1000 for key in list(strikes)})
    
    # set date as index
    sheet = sheet.set_index(sheet.mat)
    
    E = nstrikes             # strike prices
    S = sheet.s_price        # stock prices
    Cobs = sheet.iloc[:, E]  # call prices
    r = sheet.r              # risk-free interest rates
    mat = sheet.mat          # days to maturity
    T = sheet.time           # years to maturity
    
    return sheet, (mat, T, S, Cobs, E, r)

def format_sheet(sheet):
    """
    Rename columns
    """
    return sheet.rename(columns={
        sheet.columns[0]: "mat",
        sheet.columns[-3]: 's_price',
        sheet.columns[-2]: 'r',
        sheet.columns[-1]: 'date'
    })

def get_sheet(data, sheet_name):
    """
    Return cleaned sheet and destructured variables
    """
    sheet = format_sheet(data[sheet_name])
    return clean_sheet(sheet)
    
    
def save_tex(df, dest):
    """
    Save dataframe content to .tex
    """
    with open(dest, 'w') as f:
        f.write(df.to_latex())
