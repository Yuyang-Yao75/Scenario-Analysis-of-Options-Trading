# Copyright (c) 2025 Yuyang Yao
# SPDX-License-Identifier: MIT

import matplotlib
from iFinDPy import *
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.stats import norm
from typing import Dict, Hashable
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['axes.unicode_minus']=False
import os
from scipy.optimize import bisect

# 定义常量
VALID_SESSIONS = ["盘前", "盘中", "盘后"]
ANN_TRADING_DAYS=244

def validate_session(session: str) -> None:
    """
    验证交易时段参数是否有效
    
    参数:
        session: str，交易时段
    
    异常:
        ValueError: 如果session参数无效
    """
    if session not in VALID_SESSIONS:
        raise ValueError(f"无效的交易时段: {session}，有效值为: {', '.join(VALID_SESSIONS)}")

# 定义登陆函数
def ths_login():
    ret = THS_iFinDLogin('aaaaa', 'aaaaa')
    print(ret)
    if ret != 0:
        raise  RuntimeError("登陆失败")
    print("登陆成功")
#------------数据预处理-------------------
def check_and_backup_out(opt_full_post, backup_path="期权持仓表备份.csv"):
    # 先检查是否有缺失值
    has_null = np.any(opt_full_post.isnull().values)
    
    # 如果有缺失值,先保存备份
    if has_null:
        try:
            opt_full_post.to_csv(backup_path, index=False)
            print(f"检测到缺失值，当前表格已备份至:{backup_path}")
        except Exception as e:
            print(f"备份文件保存失败: {str(e)}")
            
        # 获取缺失值信息
        null_info = opt_full_post.isnull().sum()
        null_cols = null_info[null_info > 0]
        
        # 构建错误信息
        error_msg = "数据中存在缺失值，请检查以下列的数据完整性：\n"
        for col, count in null_cols.items():
            total_rows = len(opt_full_post)
            percentage = (count / total_rows) * 100
            error_msg += f"- {col}: 缺失 {count} 条数据 ({percentage:.1f}%)\n"
        
        error_msg += "\n建议：\n"
        error_msg += "1. 检查同花顺数据接口是否正常\n"
        error_msg += "2. 检查期权代码是否正确\n" 
        error_msg += "3. 检查网络连接是否稳定\n"
        error_msg += "4. 如果问题持续，可以尝试重新运行程序"
        
        # 保存完成后抛出异常
        raise ValueError(error_msg)
def fill_data(opt_post: pd.DataFrame, today: date, session: str = "盘后") -> pd.DataFrame:
    data = opt_post.copy()
    opt_list = data["代码"].tolist()
    
    # 处理日期
    if session == "盘后":
        today_str = today.strftime("%Y-%m-%d")
    else:
        today_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        
    result = THS_BD(opt_list,
                    'ths_exchange_option;ths_underlying_code_option;ths_underlying_close_option;ths_strike_price_option;ths_maturity_date_option;ths_remain_trade_days;ths_close_price_option;ths_contract_type_option;ths_contract_multiplier;ths_strike_method_option;ths_exchange_option',
                    f';;{today_str};{today_str};;{today_str},0;{today_str};;{today_str};;')
    if result.errorcode != 0:
        raise RuntimeError(f"THS_BD查询失败：{result.errmsg}")
    df = result.data.rename(columns={
        'thscode': "代码",
        'ths_underlying_code_option': "标的资产",
        'ths_underlying_close_option': "标的收盘价",
        'ths_strike_price_option': "执行价",
        'ths_maturity_date_option': "到期日",
        'ths_remain_trade_days': "剩余交易日",
        'ths_close_price_option': "期权收盘价",
        'ths_contract_type_option': "期权类型",
        'ths_contract_multiplier': "合约乘数",
        'ths_strike_method_option': "行权方式",
        'ths_exchange_option': "交易所"
    })
    data = pd.merge(data, df, on='代码', how='inner')
    check_and_backup_out(data)
    return data
def load_position_table(today, session="盘后"):
    backup_path = "../期权持仓表备份.csv"
    original_path = "../期权持仓表.csv"
    
    if os.path.exists(backup_path):
        df=pd.read_csv(backup_path,encoding="utf-8")
        # df = pd.read_csv(backup_path, encoding="gbk")
        print(f"读取：{backup_path}")
    else:
        df=pd.read_csv(original_path,encoding="gbk")
        # df = pd.read_csv(original_path, encoding="utf-8")
        print(f"读取：{original_path}")
        df = fill_data(df, today, session)
    return df
#------------隐含波动率计算----------------
def get_exchange_suffix(exchange_option):
    if exchange_option=="上期能源":
        ex_suf=".INE"
    elif exchange_option=="上海期货交易所":
        ex_suf = ".SHF"
    elif exchange_option=="上交所":
        ex_suf = ".SH"
    elif exchange_option=="深交所":
        ex_suf = ".SZ"
    elif exchange_option == "中国金融期货交易所":
        ex_suf = ".SH"
    elif exchange_option == "大连商品交易所":
        ex_suf = ".DCE"
    elif exchange_option == "郑州商品交易所":
        ex_suf = ".CZC"
    elif exchange_option == "广州期货交易所":
        ex_suf = ".GFE"
    else:
        raise ValueError
    return ex_suf
#获取上交所ETF期权整体交易情况
def get_market_overall_data(symbol: str, session: str = "盘后") -> pd.DataFrame:
    """
    获取期权市场整体交易数据
    
    参数:
        symbol: 标的代码
        session: 交易时段，可选值为"盘前"、"盘中"、"盘后"
    
    异常:
        ValueError: 如果session参数无效
    """
    validate_session(session)
    
    # 处理日期
    if session == "盘后":
        today = date.today().strftime("%Y-%m-%d")
    else:
        today = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        
    # 月合约基本信息,输入参数:开始日期(sdate)、选择日期(edate)、标的合约(bdid)、合约状态(hyzt)-iFinD数据接口
    data_df = pd.DataFrame()
    data_etfopt = THS_DR('p02653', f'sdate={today.replace("-", "")};edate={today.replace("-", "")};bdid={symbol};hyzt=0',
                        'p02653_f001:Y', 'format:dataframe')
    if data_etfopt.errorcode!=0:
        print('error:{}'.format(data_etfopt.errmsg))
    else:
        optcode_sh_list=data_etfopt.data['p02653_f001'].tolist()
        data_result=THS_BD(optcode_sh_list,
            'ths_strike_price_option;ths_remain_trade_days;ths_close_price_option;ths_contract_type_option',
            f'{today};{today},0;{today};')
        if data_result.errorcode != 0:
            print('error:{}'.format(data_result.errmsg))
        else:
            data_df = data_result.data.rename(columns={
                'thscode':"代码",
                'ths_strike_price_option': "执行价",
                'ths_remain_trade_days': "剩余交易日",
                'ths_close_price_option': "期权收盘价",
                'ths_contract_type_option':"期权类型"
            })
    return data_df
def implied_vol(code: str, symbol: str, mkt_price: float, S: float, K: float, T: float, 
                r: float, opt_type: str, ex_opt: str, session: str = "盘后", 
                sigma_init: float = 0.2, tol: float = 1e-8, max_iter: int = 10000) -> float:
    """
    计算期权的隐含波动率
    
    参数:
        code: 期权代码
        symbol: 标的代码
        mkt_price: 期权市场价格
        S: 标的资产价格
        K: 执行价格
        T: 到期时间（年）
        r: 无风险利率
        opt_type: 期权类型（"看涨期权"/"看跌期权"）
        ex_opt: 交易所名称
        session: 交易时段，可选值为"盘前"、"盘中"、"盘后"
        sigma_init: 初始波动率猜测值
        tol: 收敛容差
        max_iter: 最大迭代次数
    
    异常:
        ValueError: 如果session参数无效
    """
    validate_session(session)
    
    disc_K=K*np.exp(-r*T)
    if opt_type=="看涨期权":
        lower,upper = max(0,S-disc_K),S
    else:
        lower,upper = max(0,disc_K-S),disc_K

    if not(lower<=mkt_price<=upper):
        print(f"{code}价格越界[{lower:.4f},{upper:.4f}]")
        ex_suf=get_exchange_suffix(ex_opt)
        symbol=str(symbol)+ex_suf
        etf_data=get_market_overall_data(symbol, session)
        if opt_type == "看跌期权":
            # 查找对应的看跌期权
            put_mask = (
                    (etf_data["执行价"] == K) &
                    (etf_data["剩余交易日"] == round(T * ANN_TRADING_DAYS) + 1) &  # 更稳健的取整
                    (etf_data["期权类型"] == "看跌期权")
            )

            if np.any(put_mask):  # 检查是否有匹配项
                mkt_put = etf_data.loc[put_mask, "期权收盘价"].values[0]
                c_syn = mkt_put + S - disc_K
                mkt_price = c_syn
                print("利用平价公式调整看涨期权价格")
            else:
                raise ValueError("警告：未找到匹配的看跌期权进行平价调整")
                # 这里需要添加备用逻辑
        else:
            # 查找对应的看涨期权
            call_mask = (
                    (etf_data["执行价"] == K) &
                    (etf_data["剩余交易日"] == round(T * ANN_TRADING_DAYS) + 1) &
                    (etf_data["期权类型"] == "看涨期权")
            )

            if np.any(call_mask):
                mkt_call = etf_data.loc[call_mask, "期权收盘价"].values[0]
                p_syn = mkt_call - S + disc_K
                mkt_price = p_syn
                print("利用平价公式调整看跌期权价格")
            else:
                raise ValueError("警告：未找到匹配的看涨期权进行平价调整")

        if mkt_price<lower or mkt_price>upper:
            clipped = lower+tol if mkt_price<lower else upper-tol
            print(f"调整后仍然越界，截断价格至{clipped}")
            mkt_price=clipped

    sigma=sigma_init
    for i in range(max_iter):
        price=bs_price(S,K,T,r,sigma,opt_type)
        diff=price-mkt_price
        if abs(diff)<tol:
            return sigma
        vega=bs_vega(S,K,T,r,sigma)
        if vega<1e-8:
            print(f"{code}第{i}次迭代时vega={vega}过小，无法满足牛顿法收敛条件")
            break
        sigma-=diff/vega
        if sigma<=0:
            print(f"{code}第{i}次循环时sigma小于0，出现异常")
    print(f"{code}迭代未收敛，返回初始值sigma={sigma_init:.4f}")
    return sigma_init
#仅针对美式期货期权使用，所以设置q=r
def implied_vol_bisect(code:str,mkt_price:float,F:float,K:float,T:float,r:float,opt_type:str)->float:
    q = r
    func=lambda sigma:baw_american_option_price(F,K,T,r,sigma, opt_type,q)-mkt_price
    try:
        implied_volatility=bisect(func,0.1,1,maxiter=10000)
        return implied_volatility
    except ValueError as e:
        print(f"{code}出现ValueError{e}")
        implied_volatility=-1
        return implied_volatility
    except Exception as e:
        print(f"{code}出现其他错误{e}")
        implied_volatility=-1
        return implied_volatility
def compute_implied_vols(df: pd.DataFrame, r: float, session: str = "盘后") -> pd.DataFrame:
    """
    计算DataFrame中所有期权的隐含波动率
    
    参数:
        df: 包含期权数据的DataFrame
        r: 无风险利率
        session: 交易时段，可选值为"盘前"、"盘中"、"盘后"
    
    异常:
        ValueError: 如果session参数无效
    """
    validate_session(session)
    
    df["隐含波动率"] = df.apply(
        lambda row: implied_vol(
            row["代码"],
            row["标的资产"],
            row["期权收盘价"],
            row["平值标的价"],
            # row["标的收盘价"],
            row["执行价"],
            row["剩余交易日"],
            r,
            row["期权类型"],
            row["交易所"],
            session
        ) if row["行权方式"] == "欧式期权" else implied_vol_bisect(
            row["代码"],
            row["期权收盘价"],
            row["标的收盘价"],
            row["执行价"],
            row["剩余交易日"],
            r,
            row["期权类型"],
        ),
        axis=1
    )
    return df

def calculate_atm_underlying_price(df: pd.DataFrame, r: float, session: str) -> pd.DataFrame:
    """
    计算平值标的价
    对于美式期权返回nan，对于欧式期权通过平价公式计算
    
    参数:
        df: 包含期权数据的DataFrame
        r: 无风险利率
        session: 交易时段，可选值为"盘前"、"盘中"、"盘后"
    
    异常:
        ValueError: 如果session参数无效
    """
    validate_session(session)
    
    def get_atm_price(row, session):
        if row["行权方式"] == "美式期权":
            return np.nan
        
        try:
            # 获取标的代码和到期日
            symbol = row["标的资产"]
            ex_suf = get_exchange_suffix(row["交易所"])
            symbol = str(symbol) + ex_suf
            
            # 获取所有期权数据
            etf_data = get_market_overall_data(symbol, session)
            etf_data = etf_data[etf_data["剩余交易日"] == (row["剩余交易日"]*ANN_TRADING_DAYS+1)]
            
            # 获取标的收盘价
            underlying_price = row["标的收盘价"]
            
            # 分离看涨和看跌期权
            call_options = etf_data[etf_data["期权类型"] == "看涨期权"]
            put_options = etf_data[etf_data["期权类型"] == "看跌期权"]
            
            # 找到最接近标的收盘价的执行价
            call_strikes = call_options["执行价"].values
            put_strikes = put_options["执行价"].values
        
            # 获取平值期权及其上下两档的数据
            # 对看涨期权，找到最接近的5个执行价
            call_strikes_sorted = sorted(call_strikes, key=lambda x: abs(x - underlying_price))
            call_data = call_options[call_options["执行价"].isin(call_strikes_sorted[:5])].sort_values("执行价")
            
            # 对看跌期权，找到最接近的5个执行价
            put_strikes_sorted = sorted(put_strikes, key=lambda x: abs(x - underlying_price))
            put_data = put_options[put_options["执行价"].isin(put_strikes_sorted[:5])].sort_values("执行价")
            
            # 验证数据完整性
            if len(call_data) != 5 or len(put_data) != 5:
                print(f"警告：期权数据不完整 - 看涨期权: {len(call_data)}个, 看跌期权: {len(put_data)}个")
                return np.nan
            
            # 计算每个价位的标的价
            underlying_prices = []
            for i in range(len(call_data)):
                call_price = call_data.iloc[i]["期权收盘价"]
                put_price = put_data.iloc[i]["期权收盘价"]
                K = call_data.iloc[i]["执行价"]  # 行权价应该相同
                T = row["剩余交易日"]
                
                # 使用平价公式计算标的价
                disc_K = K * np.exp(-r * T)
                underlying_price = call_price - put_price + disc_K
                underlying_prices.append(underlying_price)
            
            # 返回平均值
            return np.mean(underlying_prices) if underlying_prices else np.nan
            
        except Exception as e:
            print(f'计算平值标的价时发生错误: {str(e)}')
            print(f'标的: {symbol}')
            return np.nan
    
    # 创建一个新的DataFrame来存储计算结果
    result_df = df.copy()
    
    # 按标的资产分组，对每个标的只计算一次平值标的价
    for symbol, group in df.groupby("标的资产"):
        # 获取该标的的第一行数据用于计算
        first_row = group.iloc[0]
        atm_price = get_atm_price(first_row, session)
        
        # 将该标的的所有行的平值标的价设置为相同的值
        result_df.loc[result_df["标的资产"] == symbol, "平值标的价"] = atm_price
    
    return result_df

#------------欧式与美式闭式定价公式----------
def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str, q:float=0.0,
            ds_list: np.ndarray = None, dv_list: np.ndarray = None) -> float or np.ndarray:
    """
    计算Black-Scholes期权价格，支持单一价格和网格价格计算

    参数:
        S: 标的资产当前价格
        K: 行权价
        T: 到期时间(年)
        r: 无风险利率
        sigma: 波动率
        opt_type: 期权类型('看涨期权'或'看跌期权')
        ds_list: 价格变化比例数组(如None则计算单一价格)
        dv_list: 波动率变化值数组(如None则计算单一价格)

    返回:
        单一价格或网格价格矩阵
    """
    # 处理输入参数，统一为数组形式
    S_arr = S * (1 + ds_list[:, None]) if ds_list is not None else S
    sigma_arr = sigma + dv_list[None, :] if dv_list is not None else sigma
    b=r-q
    # 统一计算
    d1 = (np.log(S_arr / K) + (b + 0.5 * sigma_arr ** 2) * T) / (sigma_arr * np.sqrt(T))
    d2 = d1 - sigma_arr * np.sqrt(T)

    # 根据期权类型返回结果
    if opt_type == '看涨期权':
        price = np.exp((b - r) * T)*S_arr * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - np.exp((b - r) * T)*S_arr * norm.cdf(-d1)

    # 如果输入是标量，返回标量结果
    return price

def bs_vega(S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S*norm.pdf(d1)*np.sqrt(T)

def critical_price(K, T, r, b, sigma, dv_list, opt_type):
    """
    计算美式期权临界价格的通用函数
    
    参数:
        K: 行权价
        T: 到期时间(年)
        r: 无风险利率
        b: 成本收益率 (r - q)
        sigma: 波动率
        dv_list: 波动率变化值数组(如None则计算单一价格)
        opt_type: 期权类型('看涨期权'或'看跌期权')
    
    返回:
        临界价格S_star
    """
    # 处理标量和数组的情况
    is_scalar = dv_list is None
    sigma_arr = sigma if is_scalar else sigma + dv_list[None, :]
    
    # 计算共同参数
    N = 2 * b / sigma_arr ** 2
    m = 2 * r / sigma_arr ** 2
    k = 2 * r / (sigma_arr ** 2 * (1 - np.exp(-r * T)))
    
    # 根据期权类型计算不同参数
    if opt_type == '看涨期权':
        q_u = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * m)) / 2
        S_inf = K / (1 - 1 / q_u)
        h = -(b * T + 2 * sigma_arr * np.sqrt(T)) * K / (S_inf - K)
        S0 = K + (S_inf - K) * (1 - np.exp(h))
        Q = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * k)) / 2
    else:  # 看跌期权
        q_u = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * m)) / 2
        S_inf = K / (1 - 1 / q_u)
        h = (b * T - 2 * sigma_arr * np.sqrt(T)) * K / (K - S_inf)
        S0 = S_inf + (K - S_inf) * np.exp(h)
        Q = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * k)) / 2

    # 初始化S_star和收敛状态
    S_star = float(S0) if is_scalar else S0.copy()
    not_converged = True if is_scalar else np.ones_like(S_star, dtype=bool)
    
    # 迭代求解
    max_iter = 100
    tol = 1e-5  # 收敛容差
    relax_factor = 0.8  # 松弛因子
    
    # 初始化存储中间结果的数组
    if not is_scalar:
        LHS = np.zeros_like(S_star)
        RHS = np.zeros_like(S_star)
        euro = np.zeros_like(S_star)
        d1 = np.zeros_like(S_star)
        bi = np.zeros_like(S_star)
        S_new = np.zeros_like(S_star)
        rel_error = np.zeros_like(S_star)
    
    for _ in range(max_iter):
        # 检查是否所有元素都已收敛
        if is_scalar and not not_converged:
            break
        elif not is_scalar and not np.any(not_converged):
            break
            
        if is_scalar:
            if not_converged:
                # 计算d1
                d1 = (np.log(S_star / K) + (b + 0.5 * sigma_arr ** 2) * T) / (sigma_arr * np.sqrt(T))
                
                # 计算欧式期权价格
                euro = bs_price(S_star, K, T, r, sigma_arr, opt_type, dv_list=dv_list)
                
                # 根据期权类型计算不同参数
                if opt_type == '看涨期权':
                    LHS = S_star - K
                    RHS = euro + (1 - np.exp((b - r) * T) * norm.cdf(d1)) * S_star / Q
                    bi = (np.exp((b - r) * T) * norm.cdf(d1) * (1 - 1 / Q) + 
                        (1 - np.exp((b - r) * T) * norm.pdf(d1) / (sigma_arr * np.sqrt(T))) / Q)
                    S_new = (K + RHS - bi * S_star) / (1 - bi)
                else:  # 看跌期权
                    LHS = K - S_star
                    RHS = euro - (1 - np.exp((b - r) * T) * norm.cdf(-d1)) * S_star / Q
                    bi = (-np.exp((b - r) * T) * norm.cdf(-d1) * (1 - 1 / Q) - 
                        (1 + np.exp((b - r) * T) * norm.pdf(-d1) / (sigma_arr * np.sqrt(T))) / Q)
                    S_new = (K - RHS + bi * S_star) / (1 + bi)
                
                # 计算相对误差
                rel_error = np.abs(LHS - RHS) / K
                not_converged = rel_error > tol
                
                # 更新S_star
                if not_converged:
                    S_star = relax_factor * S_new + (1 - relax_factor) * S_star
        else:
            # 只对未收敛的部分进行计算
            if np.any(not_converged):
                # 计算d1
                d1[not_converged] = (np.log(S_star[not_converged] / K) + 
                                (b + 0.5 * sigma_arr[not_converged] ** 2) * T) / (sigma_arr[not_converged] * np.sqrt(T))
                
                # 计算欧式期权价格 - 只对未收敛部分
                euro[not_converged] = bs_price(S_star[not_converged], K, T, r, sigma_arr[not_converged],
                                        opt_type, dv_list=None)  # 注意这里dv_list设为None因为我们已经在sigma_arr中考虑了

                # 根据期权类型计算不同参数
                if opt_type == '看涨期权':
                    LHS[not_converged] = S_star[not_converged] - K
                    RHS[not_converged] = (euro[not_converged] + 
                                    (1 - np.exp((b - r) * T) * norm.cdf(d1[not_converged])) *
                                    S_star[not_converged] / Q[not_converged])
                    bi[not_converged] = (np.exp((b - r) * T) * norm.cdf(d1[not_converged]) * (1 - 1 / Q[not_converged]) + 
                                    (1 - np.exp((b - r) * T) * norm.pdf(d1[not_converged]) / 
                                    (sigma_arr[not_converged] * np.sqrt(T))) / Q[not_converged])
                    S_new[not_converged] = (K + RHS[not_converged] - bi[not_converged] * S_star[not_converged]) / (1 - bi[not_converged])
                else:  # 看跌期权
                    LHS[not_converged] = K - S_star[not_converged]
                    RHS[not_converged] = (euro[not_converged] - 
                                    (1 - np.exp((b - r) * T) * norm.cdf(-d1[not_converged])) *
                                    S_star[not_converged] / Q[not_converged])
                    bi[not_converged] = (-np.exp((b - r) * T) * norm.cdf(-d1[not_converged]) * (1 - 1 / Q[not_converged]) - 
                                    (1 + np.exp((b - r) * T) * norm.pdf(-d1[not_converged]) / 
                                    (sigma_arr[not_converged] * np.sqrt(T))) / Q[not_converged])
                    S_new[not_converged] = (K - RHS[not_converged] + bi[not_converged] * S_star[not_converged]) / (1 + bi[not_converged])
                
                # 计算相对误差
                rel_error[not_converged] = np.abs(LHS[not_converged] - RHS[not_converged]) / K
                not_converged = rel_error > tol
                
                # 更新S_star
                S_star[not_converged] = relax_factor * S_new[not_converged] + (1 - relax_factor) * S_star[not_converged]
        
        # 如果达到最大迭代次数仍未收敛，打印警告
        if _ == max_iter - 1 and ((is_scalar and not_converged) or (not is_scalar and np.any(not_converged))):
            if is_scalar:
                print(f"警告：{opt_type}在{max_iter}次迭代后仍未收敛，相对误差={rel_error:.2e}")
            elif isinstance(rel_error, np.ndarray):
                max_error = np.max(rel_error[not_converged])
                print(f"警告：{opt_type}部分元素在{max_iter}次迭代后仍未收敛，最大相对误差={max_error:.2e}，sigma为{sigma_arr[not_converged]}，值为{S_star[not_converged]}")
            else:
                print(f"警告：{opt_type}部分元素在{max_iter}次迭代后仍未收敛")
    
    return S_star

# 为了保持兼容性，保留原始函数作为包装器
def critical_price_call(K, T, r, b, sigma, dv_list):
    return critical_price(K, T, r, b, sigma, dv_list, '看涨期权')

def critical_price_put(K, T, r, b, sigma, dv_list):
    return critical_price(K, T, r, b, sigma, dv_list, '看跌期权')

def baw_american_option_price(F,K,T,r,sigma,opt_type,q=0.0,
                            ds_list:np.ndarray=None,dv_list:np.ndarray=None,code:str='00000.00',mkt_value:float=0.0):
    b=r-q
    if sigma<=0:
        #该if存在一点问题，如果在后续修改过程中出现单一值输入sigma小于零无法应对，
        print(f"{code}隐含波动率计算出现错误，为消除影响所有情景均按市值计算")
        return np.full((len(ds_list),len(dv_list)),mkt_value) if (ds_list is not None and dv_list is not None) else mkt_value
    if opt_type == '看涨期权':
        if b >= r:
            return bs_price(F, K, T, r, sigma, '看涨期权',q=q, ds_list=ds_list,dv_list=dv_list)
        else:
            F_arr = (1 + ds_list[:, None]) * F if ds_list is not None else F
            sigma_arr = sigma + dv_list[None, :] if dv_list is not None else sigma
            F_star = critical_price_call(K, T, r, b, sigma,dv_list)#是一个完整的11*11的array
            d1_star = (np.log(F_star / K) + (b + 0.5 * sigma_arr ** 2) * T) / (sigma_arr * np.sqrt(T))
            N = 2 * b / sigma_arr ** 2
            k = 2 * r / (sigma_arr ** 2 * (1 - np.exp(-r * T)))
            Q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * k)) / 2
            A2 = (F_star / Q2) * (1 - np.exp((b - r) * T) * norm.cdf(d1_star))
            euro_vals=bs_price(F, K, T, r, sigma, '看涨期权',q=q,ds_list=ds_list,dv_list=dv_list)
            return np.where(F_arr<F_star,euro_vals+A2*(F_arr/F_star)**Q2,F_arr-K)
    elif opt_type == '看跌期权':
        F_arr = (1 + ds_list[:, None]) * F if ds_list is not None else F
        sigma_arr = sigma + dv_list[None, :] if dv_list is not None else sigma
        F_star = critical_price_put(K, T, r, b, sigma,dv_list)
        d1_star = (np.log(F_star / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        N = 2 * b / sigma_arr ** 2
        k = 2 * r / (sigma_arr ** 2 * (1 - np.exp(-r * T)))
        Q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * k)) / 2
        A1 = -(F_star / Q1) * (1 - np.exp((b - r) * T) * norm.cdf(-d1_star))
        euro_vals = bs_price(F, K, T, r, sigma, '看跌期权',q=q, ds_list=ds_list, dv_list=dv_list)
        return np.where(F_arr>F_star,euro_vals+A1*(F_arr/F_star)**Q1,K-F_arr)
    else:
        raise ValueError("opt_type must be 'call' or 'put'")
#------------情景分析---------------------
def generate_grid(ds_steps=11, dv_steps=11):
    ds_list = np.round(np.linspace(-0.1, 0.1, ds_steps), 2)  # 保留2位小数
    dv_list = np.round(np.linspace(-0.1, 0.1, dv_steps), 2)  # 保留2位小数
    return ds_list, dv_list
#------------主函数-----------------------
def calculate_total_margin(df: pd.DataFrame, ds_list: np.ndarray) -> pd.DataFrame:
    """
    计算期权的保证金，支持情景分析
    
    参数:
        df: 包含期权数据的DataFrame
        ds_list: 标的资产价格变动比例数组
    
    返回:
        包含每个合约在不同情景（标的资产变化）下的保证金计算结果的DataFrame
    """
    def calculate_margin(row: pd.Series, ds: float) -> float:
        # 计算调整后的标的资产价格
        S = row['标的收盘价'] * (1 + ds)
        K = row['执行价']
        opt_price = row['期权收盘价']
        multiplier = row['合约乘数']
        quantity = row['数量']
        
        # 多头持仓保证金为0
        if quantity >=0:
            return 0.0
            
        # 计算虚值额
        if row['期权类型'] == '看涨期权':
            out_of_money = max(K - S, 0) 
        else:  # 看跌期权
            out_of_money = max(S - K, 0) 
            
        # 根据不同交易所计算保证金
        if row['交易所'] in ['上交所', '深交所']:
            margin = (opt_price + max(0.12 * S - out_of_money, 0.07 * S)) * multiplier
        elif row['交易所'] == '中国金融期货交易所':
            margin = (opt_price + max(0.12 * S - out_of_money, 0.06 * S)) * multiplier
        else:  # 其他交易所（期货交易所）
            # 计算标的期货合约交易保证金
            underlying=row['标的资产']+get_exchange_suffix(row['交易所'])
            # 这里无论盘前盘中还是盘后都是用距离最近的结算价格来算的，所以日期出不用做特殊调整；另外只有期权空头才要交保证金，所以直接使用期货空头保证金计算
            futures_margin_rate = THS_BD(underlying,'ths_contract_short_deposit_future',f'{datetime.today().strftime("%Y-%m-%d")}').data['ths_contract_short_deposit_future'].values[0]
            futures_margin=S*futures_margin_rate/100
            margin = max(
                opt_price + futures_margin - 0.5 * out_of_money,
                opt_price + 0.5 * futures_margin
            )*multiplier
            
        return margin * abs(quantity)
    
    # 创建结果DataFrame
    result = pd.DataFrame(index=df.index, columns=[f"{int(x*100):+d}%" for x in ds_list])
    
    # 对每个价格变动情景计算保证金
    for ds in ds_list:
        result[f"{int(ds*100):+d}%"] = df.apply(lambda row: calculate_margin(row, ds), axis=1)
    
    return result

def calculate_greeks(df: pd.DataFrame, ds_list: np.ndarray, dv_list: np.ndarray, r: float) -> Dict[Hashable, Dict[str, pd.DataFrame]]:
    """
    计算期权的希腊值，支持情景分析
    
    参数:
        df: 包含期权数据的DataFrame
        ds_list: 标的资产价格变动比例数组
        dv_list: 波动率变动比例数组
        r: 无风险利率
    
    返回:
        包含每个合约在不同情景下的希腊值计算结果的DataFrame
    """
    def calculate_european_greeks(row: pd.Series, ds: float, dv: float) -> dict:#针对单个数的计算
        """计算欧式期权的希腊值（解析解）"""
        S = row['标的收盘价'] * (1 + ds)
        K = row['执行价']
        T = row['剩余交易日']
        sigma = row['隐含波动率'] * (1 + dv)
        multiplier = row['合约乘数']
        quantity = row['数量']
        q = 0  # 假设无股息
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if row['期权类型'] == '看涨期权':
            delta = np.exp(-q * T) * norm.cdf(d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # 除以100转换为每1%波动率变化
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / ANN_TRADING_DAYS  # 转换为每日theta #todo是不是要转化成日化的
        else:  # 看跌期权
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / ANN_TRADING_DAYS

        return {
            'delta': 0.01*S*delta*multiplier*quantity,
            'gamma': 0.5*0.01**2*S**2*gamma*multiplier*quantity,
            'vega': vega*multiplier*quantity,
            'theta': theta*multiplier*quantity
        }
    
    def calculate_american_greeks(row: pd.Series, ds: float, dv: float) -> dict:
        """计算美式期权的希腊值（差分法）"""
        S = row['标的收盘价'] * (1 + ds)
        K = row['执行价']
        T = row['剩余交易日']
        sigma = row['隐含波动率'] * (1 + dv)
        q = r  # 美式期货期权，q = r
        multiplier = row['合约乘数']
        quantity = row['数量']
        # 设置差分步长
        dS = S * 0.0001  # 万分之一的价格变动
        dsigma = sigma * 0.0001  # 万分之一的波动率变动
        dT = 1/ANN_TRADING_DAYS  # 1天的时间变动
        
        # 计算基础价格
        base_price = baw_american_option_price(S, K, T, r, sigma, row['期权类型'], q)
        
        # 计算delta和gamma
        price_up = baw_american_option_price(S + dS, K, T, r, sigma, row['期权类型'], q)
        price_down = baw_american_option_price(S - dS, K, T, r, sigma, row['期权类型'], q)
        delta = (price_up - price_down) / (2 * dS)
        gamma = (price_up + price_down - 2 * base_price) / (dS ** 2)
        
        # 计算vega
        price_vol_up = baw_american_option_price(S, K, T, r, sigma + dsigma, row['期权类型'], q)
        price_vol_down = baw_american_option_price(S, K, T, r, sigma - dsigma, row['期权类型'], q)
        vega = (price_vol_up - price_vol_down) / (2 * dsigma * 100)   # 转换为每1%波动率变化

        # 计算theta
        price_time = baw_american_option_price(S, K, T - dT, r, sigma, row['期权类型'], q)
        theta = (price_time - base_price)
        
        return {
            'delta': 0.1*delta*multiplier*quantity*S,
            'gamma': 0.5*0.01**2*S**2*gamma*multiplier*quantity,
            'vega': vega*multiplier*quantity,
            'theta': theta*multiplier*quantity
        }

    # 初始化结果字典
    result_dict = {}
    greek_names = ['delta', 'gamma', 'vega', 'theta']

    # 按标的分组处理
    for underlying, group in df.groupby('标的资产'):
        # 初始化当前标的的希腊值DataFrame
        greek_data = {
            greek: pd.DataFrame(
                index=[f"{int(ds * 100):+d}%" for ds in ds_list],
                columns=[f"{int(dv * 100):+d}vol" for dv in dv_list]
            ) for greek in greek_names
        }

        # 对每个情景组合计算希腊值并填充
        for ds in ds_list:
            for dv in dv_list:
                # 计算当前情景下所有合约的希腊值
                temp_greeks = {greek: 0 for greek in greek_names}

                for _, row in group.iterrows():
                    if row['行权方式'] == '欧式期权':
                        greeks = calculate_european_greeks(row, ds, dv)
                    else:
                        greeks = calculate_american_greeks(row, ds, dv)

                    # 累加希腊值
                    for greek in greek_names:
                        temp_greeks[greek] += greeks[greek]

                # 填充到DataFrame中
                row_name = f"{int(ds * 100):+d}%"
                col_name = f"{int(dv * 100):+d}vol"
                for greek in greek_names:
                    greek_data[greek].loc[row_name, col_name] = temp_greeks[greek]

        result_dict[underlying] = greek_data

    return result_dict

def main():
    #设置无风险收益率
    r=0.02
    #该程序在每日盘后进行测试，因此获取当日日期参数。注：如果盘前测试，需要修改fill_data函数中收盘价涉及到的两个参数，否则无法读取数据。
    today=date.today()
    folder=today.strftime("%Y-%m-%d")
    # 运行结果保存在以日期命名的文件夹中，所有情景分析结论均放置在同一个excel表格里。
    os.makedirs(folder,exist_ok=True)
    excel_filename=os.path.join(folder,f"{folder}_情景分析.xlsx")

    with pd.ExcelWriter(excel_filename,engine='xlsxwriter') as writer:
        ths_login()
        #需要提前将"期权持仓表"保存在文件夹中，第一列为期权代码（列名"代码"，需要与同花顺代码一致），第二列为持仓数量（列名"数量"）
        #扩充表格，使其包含情景分析所必要的数据，数据缺失时报错，需人为补全数据
        #load_position_table函数有一个可选参数，可以输入盘前、盘中和盘后，盘后调取数据时日期参数为当天，盘前和盘中为昨天
        SESSION="盘后"
        opt_full_post=load_position_table(today,SESSION)
        opt_full_post.to_csv("期权持仓表备份.csv", index=False)
        opt_full_post["剩余交易日"] = (opt_full_post["剩余交易日"]-1) / ANN_TRADING_DAYS#剩余交易日需要减去一天，因为同花顺数据是包含当天的。
        opt_full_post=calculate_atm_underlying_price(opt_full_post,r,SESSION)
        opt_full_post=compute_implied_vols(opt_full_post,r,SESSION)
        opt_full_post.to_excel(writer, sheet_name="期权持仓表完整", index=False)

        ds_list, dv_list=generate_grid()
        spot_list=[f"{int(x*100):+d}%" for x in ds_list]
        vol_list=[f"{int(x*100):+d}vol" for x in dv_list]
        scenario={}

        opt_full_post["剩余交易日"] = (opt_full_post["剩余交易日"]) -1/ ANN_TRADING_DAYS#在做的是明天的情景分析，因此剩余交易日需要减去一天。
        for _,row in opt_full_post.iterrows():
            base_price=row["期权收盘价"]
            if row["行权方式"]=="欧式期权":
                grid_price=bs_price(
                    row["平值标的价"],
                    # row["标的收盘价"],
                    row["执行价"],
                    row["剩余交易日"],
                    r,
                    row["隐含波动率"],
                    row["期权类型"],
                    0,
                    ds_list,
                    dv_list,
                )
            elif row["行权方式"]=="美式期权":
                q=r
                grid_price=baw_american_option_price(
                    row["标的收盘价"],
                    row["执行价"],
                    row["剩余交易日"],
                    r,
                    row["隐含波动率"],
                    row["期权类型"],
                    q,
                    ds_list,
                    dv_list,
                    row["代码"],
                    row["期权收盘价"]
                )
            pnl=(grid_price-base_price)*row["数量"]*row["合约乘数"]
            pnl_df = pd.DataFrame(pnl, index=spot_list, columns=vol_list).round(1)
            ug=row["标的资产"]
            code=row["代码"]
            scenario.setdefault(ug,[]).append((code,pnl_df))

        grand_total=None
        for underlying,lst in scenario.items():
            total=sum(mat for _,mat in lst)
            total.to_excel(writer,sheet_name=f"{underlying}",float_format="%.4f")
            grand_total=total if grand_total is None else grand_total+total
        grand_total.to_excel(writer,sheet_name="all_scenario",float_format="%.4f")

        # 计算保证金情景分析
        margin_df = calculate_total_margin(opt_full_post, ds_list)
        
        # 按标的资产分组计算保证金
        margin_by_underlying = margin_df.groupby(opt_full_post['标的资产']).sum()
        
        # 将保证金计算结果添加到每个标的的sheet中
        for underlying in scenario.keys():
            worksheet = writer.sheets[underlying]
            # 在情景分析结果下方添加保证金信息
            row_num = len(spot_list) + 3  # 情景分析结果下方空两行
            worksheet.write(row_num, 0, f"{underlying}保证金情景分析：")
            
            # 写入保证金情景分析结果（垂直格式）
            for i, (scenario_flag, value) in enumerate(zip(margin_by_underlying.columns[::-1], margin_by_underlying.loc[underlying][::-1])):
                worksheet.write(row_num + 1 + i, 0, scenario_flag)
                worksheet.write(row_num + 1 + i, 1, value)
        
        # 在all_scenario sheet中添加总保证金
        worksheet = writer.sheets['all_scenario']
        row_num = len(spot_list) + 3
        worksheet.write(row_num, 0, "总保证金情景分析：")
        
        # 写入总保证金情景分析结果（垂直格式）
        total_margin = margin_by_underlying.sum()
        for i, (scenario_flag, value) in enumerate(zip(total_margin.index[::-1], total_margin[::-1])):
            worksheet.write(row_num + 1 + i, 0, scenario_flag)
            worksheet.write(row_num + 1 + i, 1, value)

        # 计算希腊值 - 现在返回的是嵌套字典
        # Delta*100=Wind，Gamma*200=Wind,Vega和Theta和Wind一致。
        greeks_by_underlying = calculate_greeks(opt_full_post, ds_list, dv_list, r)

        # 将希腊值结果写入Excel（每个标的单独sheet）
        for underlying in scenario.keys():
            worksheet = writer.sheets[underlying]
            row_num = len(spot_list) + len(margin_by_underlying.columns) + 5  # 预留位置

            for greek in ['delta', 'gamma', 'vega', 'theta']:
                # 写入标题
                worksheet.write(row_num, 0, f"{underlying} {greek.upper()}分析：")
                row_num += 1

                # 写入表头（波动率变动）
                for j, vol in enumerate(dv_list):
                    worksheet.write(row_num, j + 1, f"{int(vol * 100):+d}vol")

                # 写入数据（价格变动 vs 波动率变动）
                greek_matrix = greeks_by_underlying[underlying][greek]
                for i, spot in enumerate(ds_list):
                    worksheet.write(row_num + i + 1, 0, f"{int(spot * 100):+d}%")  # 行标签
                    for j, vol in enumerate(dv_list):
                        worksheet.write(row_num + i + 1, j + 1, round(greek_matrix.iloc[i, j],4))

                row_num += len(ds_list) + 2  # 跳转到下一个希腊值的位置

        # 计算并写入总希腊值（all_scenario sheet）
        if 'all_scenario' in writer.sheets:
            worksheet = writer.sheets['all_scenario']
            row_num = len(ds_list) + len(margin_by_underlying.columns) + 5
            # 初始化总希腊值矩阵
            total_greeks = {greek: pd.DataFrame(0, index=[f"{int(ds * 100):+d}%" for ds in ds_list], columns=[f"{int(dv * 100):+d}vol" for dv in dv_list])
                            for greek in ['delta', 'gamma', 'vega', 'theta']}

            # 累加所有标的的希腊值
            for underlying_data in greeks_by_underlying.values():
                for greek in total_greeks:
                    total_greeks[greek] = total_greeks[greek]+underlying_data[greek]

            # 写入Excel
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                worksheet.write(row_num, 0, f"总{greek.upper()}分析：")
                row_num += 1

                # 写入表头和数据（与前面逻辑相同）
                for j, vol in enumerate(dv_list):
                    worksheet.write(row_num, j + 1, f"{int(vol * 100):+d}vol")

                for i, spot in enumerate(ds_list):
                    worksheet.write(row_num + i + 1, 0, f"{int(spot * 100):+d}%")
                    for j, vol in enumerate(dv_list):
                        worksheet.write(row_num + i + 1, j + 1, round(total_greeks[greek].iloc[i, j],4))

                row_num += len(ds_list) + 2

        # 添加斜线表头
        workbook = writer.book
        for sheet_name,worksheet in writer.sheets.items():
            if sheet_name=='期权持仓表完整':
                continue
            diag_fmt=workbook.add_format({
                'border':1,
                'diag_type':2,
                'diag_border':1,
                'align':'left',
                'valign':'bottom',
                'text_wrap':True,
            })
            worksheet.write(0,0,'     波动率（iv）\n标的资产（S）',diag_fmt)
            worksheet.set_row(0,30)
            worksheet.set_column(0,0,17)


if __name__=="__main__":
    main()


