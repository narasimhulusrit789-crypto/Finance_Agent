"""
Sample financial knowledge documents for pre-loading into the RAG vector index.
This module seeds the knowledge base with foundational investment content.
"""

from __future__ import annotations

FINANCIAL_DOCUMENTS = {
    "investment_fundamentals.txt": """
IBM watsonx Finance Agent — Investment Knowledge Base

INVESTMENT FUNDAMENTALS
========================

1. ASSET ALLOCATION THEORY
Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952, demonstrates that 
a diversified portfolio can achieve superior risk-adjusted returns. The efficient frontier 
represents the set of optimal portfolios offering maximum expected return for a given level 
of risk.

Key allocation principles:
- Diversification reduces unsystematic (company-specific) risk
- Asset classes with low correlation reduce portfolio volatility
- Risk tolerance and investment horizon determine optimal allocation
- Rebalancing maintains target allocation as markets move

2. RISK METRICS GLOSSARY
- Beta: Measures portfolio sensitivity relative to the market (S&P 500 = 1.0)
- Volatility (Std Dev): Annualised standard deviation of returns
- Sharpe Ratio: Risk-adjusted return = (Portfolio Return - Risk-Free Rate) / Volatility
- Maximum Drawdown: Largest peak-to-trough decline in portfolio value
- Value at Risk (VaR): Maximum expected loss at a confidence level over a time period

3. ASSET CLASS CHARACTERISTICS

EQUITIES
- Historical return: 10-12% annual (S&P 500 long-term)
- Risk: High volatility, cyclical
- Best for: Long-term wealth growth (5+ year horizon)
- Key instruments: ETFs (SPY, QQQ, VTI), individual stocks, sector funds

FIXED INCOME (BONDS)
- Historical return: 3-6% annual
- Risk: Low-medium; interest rate risk, credit risk
- Best for: Income generation, capital preservation, diversification
- Key instruments: Treasury bonds, corporate bonds, bond ETFs (BND, AGG, LQD)

REAL ESTATE (REITs)
- Historical return: 8-12% total return (income + appreciation)
- Risk: Moderate; interest rate sensitive, illiquidity
- Best for: Income, inflation hedge, diversification
- Key instruments: VNQ, SCHH, real estate ETFs

COMMODITIES
- Historical return: 2-5% (varies significantly)
- Risk: High volatility, no income generation
- Best for: Inflation hedge, portfolio diversification
- Key instruments: GLD (Gold), DJP (Diversified), USO (Oil)

CASH / MONEY MARKET
- Return: 4-5% (current high-rate environment)
- Risk: Very low; inflation risk over time
- Best for: Emergency fund, short-term goals, dry powder
- Key instruments: SGOV (T-bills), money market funds, HY savings

4. RISK TOLERANCE FRAMEWORK

LOW RISK PROFILE
- Target: Capital preservation with modest growth
- Allocation: 50% Bonds, 25% Equities, 15% Cash, 10% Real Assets
- Expected return: 4-6% annually
- Suitable for: Near retirees, short horizon (<3 years), conservative investors

MEDIUM RISK PROFILE
- Target: Balanced growth and income
- Allocation: 55% Equities, 25% Bonds, 10% Real Estate, 10% Alternatives
- Expected return: 8-10% annually
- Suitable for: Mid-career investors, 5-15 year horizon, moderate temperament

HIGH RISK PROFILE
- Target: Maximum growth
- Allocation: 75% Equities, 10% International, 10% Alternatives, 5% Real Estate
- Expected return: 12-15% annually
- Suitable for: Young investors, 15+ year horizon, high loss tolerance

5. TAX-EFFICIENT INVESTING
- Maximize tax-advantaged accounts first: 401(k), IRA, Roth IRA
- Hold tax-inefficient assets (bonds, REITs) in tax-deferred accounts
- Hold tax-efficient assets (index ETFs) in taxable accounts
- Tax-loss harvesting: Offset gains with strategic loss realization
- Hold period: >12 months for long-term capital gains rate (15-20%)

6. INVESTMENT COSTS MATTER
- Expense ratios: Choose low-cost index funds (0.03-0.20%)
- Transaction costs: Use commission-free brokers
- Tax drag: Minimize unnecessary turnover
- Advisor fees: Typically 0.25-1% AUM annually
- Impact: A 1% cost difference on $100K over 30 years = ~$174,000 in lost value
""",

    "market_outlook_2024.txt": """
MARKET OUTLOOK 2024-2025
=========================

MACROECONOMIC ENVIRONMENT
- Federal Reserve: Rate cuts expected H2 2024; Fed Funds Rate 5.25-5.50%
- Inflation: CPI trending down toward 2% target; core services sticky
- GDP Growth: US economy resilient at 2.5% growth; soft landing likely
- Employment: Unemployment near historic lows at 3.7%
- Dollar: Gradual weakening expected as Fed pivots

EQUITY MARKET OUTLOOK
- Valuation: S&P 500 P/E ratio ~22x — elevated but supported by earnings growth
- Earnings: Corporate earnings expected to grow 8-10% in 2024
- AI/Technology: Secular growth theme; significant capital expenditure cycle
- Small-cap: Potential outperformance if rates decline
- International: Emerging markets attractive on valuation; currency risk
- Sectors to watch: Technology, Healthcare, Financials (rate-sensitive)

FIXED INCOME OUTLOOK  
- Duration: Extending duration makes sense as rates peak
- High-yield: Spreads remain tight; selective approach warranted
- EM debt: Attractive yields; currency risk management important
- TIPS: Inflation protection less critical as CPI normalizes
- Recommendation: Intermediate-duration investment-grade bonds

REAL ESTATE
- REITs: Rate-sensitive but fundamentals improving
- Commercial real estate: Office sector stress; industrial/logistics strong
- Residential: Supply constrained, mortgage rates easing
- Recommendation: Selective REIT exposure; avoid office-heavy funds

COMMODITIES
- Gold: Bullish; rate cuts, central bank buying, geopolitical demand
- Oil: Geopolitical risk premium; OPEC+ supply management
- Agricultural: Weather risk; climate change increasing volatility
- Recommendation: Gold ETF as inflation/tail-risk hedge (3-5% allocation)

RISK FACTORS TO MONITOR
1. Geopolitical tensions: Middle East, Taiwan, Ukraine
2. Credit cycle: High-yield defaults rising from low base
3. Commercial real estate: $1.5T refinancing wall 2024-2026
4. AI bubble risk: High valuations in AI infrastructure plays
5. Election year: Policy uncertainty in US and globally
6. China slowdown: Property sector stress, deflationary pressures

INVESTMENT STRATEGY THEMES 2024
- Quality over Growth: Focus on profitable companies with pricing power
- Duration management: Gradually extend bond duration
- International diversification: Europe, India, Japan opportunities
- AI infrastructure: Long-term secular growth play (with valuation discipline)
- Dividend growth: Quality income in uncertain environment
""",

    "portfolio_strategies.txt": """
ADVANCED PORTFOLIO STRATEGIES
==============================

1. DOLLAR-COST AVERAGING (DCA)
Strategy: Invest fixed amount at regular intervals regardless of price
Benefits:
- Reduces timing risk and emotional decision-making
- Automatically buys more shares when prices are low
- Smooths entry cost over time
Best for: Regular investors with monthly income to invest
Example: $500/month into SPY over 10 years

2. CORE-SATELLITE PORTFOLIO CONSTRUCTION
Core (70-80%): Low-cost broad market index funds
- US Total Market: VTI or ITOT
- International: VXUS or IXUS
- Bonds: BND or AGG
Satellite (20-30%): Active bets, sector tilts, thematic plays
- Technology: QQQ or XLK
- Healthcare: XLV or VHT
- Emerging Markets: EEM or VWO
- Thematic: ARKK, HACK, BOTZ

3. FACTOR INVESTING (SMART BETA)
Value Factor: Low P/E, P/B stocks historically outperform — VTV, VONV
Quality Factor: High ROE, low debt, stable earnings — QUAL, SPHQ
Momentum Factor: Recent winners tend to continue — MTUM, PDP
Low Volatility Factor: Lower vol stocks with better risk-adjusted returns — SPLV, USMV
Small Cap Value: Historically highest long-term returns — VBR, AVUV

4. INCOME INVESTING
Dividend Growth Stocks: Companies with consistent dividend increases — SCHD, VIG
REITs: Real estate income with liquidity — VNQ, O, MAIN
Bond Laddering: Buy bonds maturing in 1, 2, 3, 4, 5 years
Covered Calls: Generate income on existing equity positions
BDCs/MLPs: Higher yield alternatives with specific tax treatment

5. REBALANCING STRATEGIES
Calendar Rebalancing: Quarterly or annual
Threshold Rebalancing: Rebalance when allocation drifts >5% from target
Tax-Loss Harvesting Integration: Rebalance strategically to capture tax losses
Contribution-Based: Direct new investments to underweight assets

6. RISK MANAGEMENT TECHNIQUES
Portfolio Diversification: Max single position <10%; max sector <25%
Stop-Loss Orders: Limit downside on individual positions (15-20%)
Options Hedging: Protective puts for tail-risk events
Gold Allocation: 3-5% as portfolio insurance
Cash Buffer: 5-15% for opportunities and emotional stability
"""
}


def seed_knowledge_base(ingestion_pipeline) -> dict[str, int]:
    """Seed the vector store with foundational financial knowledge."""
    results = {}
    for doc_name, content in FINANCIAL_DOCUMENTS.items():
        try:
            n = ingestion_pipeline.ingest_text(content, source=doc_name)
            results[doc_name] = n
        except Exception as exc:
            results[doc_name] = -1
    return results
