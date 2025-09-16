"""
pta_nodal_app.py

Requirements:
    pip install streamlit pandas numpy plotly scipy openpyxl

Run:
    streamlit run pta_nodal_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from math import pi, log
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(layout="wide", page_title="PTA & Nodal Analysis", initial_sidebar_state="expanded")
st.title("PTA (Horner & Derivative)  +  Nodal Analysis (IPR vs TPR)")

# --------------------------
# Helper functions
# --------------------------



def dp_friction_per_ft(q, v, ID_inch, roughness_ft, rho_lb_ft3, mu_tpr, MD):
    # q: flow rate (STB/day) [not used directly], v: velocity (ft/s), ID_inch: tubing ID (in),
    # roughness_ft: abs roughness (ft), rho_lb_ft3: fluid density (lb/ft3), mu_tpr: viscosity (cp), MD: measured depth (ft)
    # Returns dp_per_ft (psi/ft)
    ID_ft = ID_inch / 12.0
    eps_rel = roughness_ft / ID_ft
    # Calculate Reynolds number (Re)
    # mu_tpr in cp, convert to lb/(ft·s): 1 cp = 0.000672 lb/(ft·s)
    mu_lbfts = mu_tpr * 0.000672
    Re = (rho_lb_ft3 * v * ID_ft) / mu_lbfts
    # Friction factor
    if Re < 2000:
        f = 64.0 / Re if Re > 0 else 0.02
    else:
        # Haaland equation
        try:
            f = (-1.8 * np.log10( (6.9/Re) + ( (eps_rel/3.7)**1.11 ) ))**-2
        except Exception:
            f = 0.02
    # Darcy-Weisbach equation for pressure drop per foot (lb/ft^2 per ft)
    g_c = 32.174  # lb·ft/(lbf·s^2)
    dp_per_ft_lbf_ft2 = f * rho_lb_ft3 * v**2 / (2 * g_c * ID_ft)
    # Convert to psi/ft
    dp_per_ft_psi = dp_per_ft_lbf_ft2 / 144
    return dp_per_ft_psi
# --------------------------
# Tabs: PTA and Nodal
# --------------------------
tab_pta, tab_nodal, tab_decline = st.tabs(["PTA (Horner & Derivative)", "Nodal Analysis (IPR & TPR)", "Decline Curve Analysis"])
# --------------------------
# DECLINE CURVE TAB
# --------------------------
with tab_decline:
    st.header("Decline Curve Analysis")
    st.write("Upload production data (dates and rates) to perform decline curve analysis.")
    dca_file = st.file_uploader("Upload production data (Excel/CSV) with columns 'date' and 'rate'",
                                type=["xlsx","csv"], key="dca_uploader")
    if dca_file is None:
        st.info("Upload a production data file to begin.")
    else:
        # Load data
        if dca_file.name.endswith(".xlsx"):
            df_dca = pd.read_excel(dca_file)
        else:
            df_dca = pd.read_csv(dca_file)
        st.subheader("Raw Production Data")
        st.dataframe(df_dca)
        expected = {"date", "rate"}
        if not expected.issubset(df_dca.columns):
            st.error(f"File must contain columns: {expected}")
        else:
            # Convert date column to datetime
            df_dca["date"] = pd.to_datetime(df_dca["date"])
            df_dca = df_dca.sort_values("date")
            # Allow user to select start point for analysis
            st.subheader("Select Start Point for Decline Analysis")
            idx_start = st.slider("Start index", min_value=0, max_value=len(df_dca)-2, value=0)
            df_dca_analysis = df_dca.iloc[idx_start:].reset_index(drop=True)
            st.write(f"Analysis will use data from {df_dca_analysis['date'].iloc[0].date()} onward.")
            # Choose decline methods (multi-select)
            st.subheader("Choose Decline Method(s)")
            decline_methods = st.multiselect(
                "Select decline method(s) to fit and plot:",
                ["Exponential", "Harmonic", "Hyperbolic"],
                default=["Exponential"]
            )
            # Forecast end date input
            st.subheader("Forecast End Date")
            min_date = df_dca_analysis["date"].iloc[0]
            max_date = df_dca_analysis["date"].iloc[-1] + pd.Timedelta(days=365)
            forecast_end = st.date_input("Select forecast end date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
            # Prepare data for fitting
            t_days = (df_dca_analysis["date"] - df_dca_analysis["date"].iloc[0]).dt.days.astype(float).to_numpy()
            q_obs = df_dca_analysis["rate"].to_numpy(dtype=float)
            # Forecast time grid
            forecast_end_days = (pd.to_datetime(forecast_end) - df_dca_analysis["date"].iloc[0]).days
            t_forecast = np.arange(0, forecast_end_days + 1)
            date_forecast = df_dca_analysis["date"].iloc[0] + pd.to_timedelta(t_forecast, unit="D")
            results = []
            import plotly.graph_objects as go
            fig_dca = go.Figure()
            fig_dca.add_trace(go.Scatter(x=df_dca_analysis["date"], y=q_obs, mode="markers+lines", name="Observed Rates",
                                         marker=dict(size=8, color="#1f77b4"), line=dict(width=2, color="#1f77b4"), yaxis="y1"))
            # Prepare for cumulative production plot
            cumu_traces = []
            # Fit and plot each selected method
            from scipy.optimize import curve_fit
            def exp_decline(t, qi, D):
                return qi * np.exp(-D * t)
            def harm_decline(t, qi, D):
                return qi / (1 + D * t)
            def hyp_decline(t, qi, D, b):
                return qi / (1 + b * D * t) ** (1 / b)
            t_fit = t_days
            for method in decline_methods:
                if method == "Exponential":
                    try:
                        popt, _ = curve_fit(exp_decline, t_fit, q_obs, p0=[q_obs[0], 0.001], maxfev=10000)
                        qi, D = popt
                        q_fit = exp_decline(t_forecast, qi, D)
                        fig_dca.add_trace(go.Scatter(x=date_forecast, y=q_fit, mode="lines", name="Exponential",
                                                     line=dict(width=3, dash="dash", color="#d62728"), yaxis="y1"))
                        # Cumulative production (sum of daily rates)
                        cum_fit = np.cumsum(q_fit)
                        cumu_traces.append(go.Scatter(x=date_forecast, y=cum_fit, mode="lines", name="Cumulative Exponential",
                                                     line=dict(width=2, dash="dash", color="#d62728"), yaxis="y2"))
                        # EUR for exponential: EUR = qi/D * (1 - exp(-D * t_end)), t_end in days
                        t_end = t_forecast[-1]
                        EUR = qi / D * (1 - np.exp(-D * t_end)) if D > 0 else np.nan
                        results.append({"Method": "Exponential", "qi": qi, "D": D, "b": np.nan, "EUR": EUR})
                    except Exception:
                        results.append({"Method": "Exponential", "qi": np.nan, "D": np.nan, "b": np.nan, "EUR": np.nan})
                elif method == "Harmonic":
                    try:
                        popt, _ = curve_fit(harm_decline, t_fit, q_obs, p0=[q_obs[0], 0.001], maxfev=10000)
                        qi, D = popt
                        q_fit = harm_decline(t_forecast, qi, D)
                        fig_dca.add_trace(go.Scatter(x=date_forecast, y=q_fit, mode="lines", name="Harmonic",
                                                     line=dict(width=3, dash="dot", color="#2ca02c"), yaxis="y1"))
                        cum_fit = np.cumsum(q_fit)
                        cumu_traces.append(go.Scatter(x=date_forecast, y=cum_fit, mode="lines", name="Cumulative Harmonic",
                                                     line=dict(width=2, dash="dot", color="#2ca02c"), yaxis="y2"))
                        # EUR for harmonic: EUR = qi/D * ln(1 + D * t_end)
                        t_end = t_forecast[-1]
                        EUR = qi / D * np.log(1 + D * t_end) if D > 0 else np.nan
                        results.append({"Method": "Harmonic", "qi": qi, "D": D, "b": np.nan, "EUR": EUR})
                    except Exception:
                        results.append({"Method": "Harmonic", "qi": np.nan, "D": np.nan, "b": np.nan, "EUR": np.nan})
                elif method == "Hyperbolic":
                    try:
                        popt, _ = curve_fit(hyp_decline, t_fit, q_obs, p0=[q_obs[0], 0.001, 0.5], bounds=([0,0,0],[np.inf,1,2]), maxfev=10000)
                        qi, D, b = popt
                        q_fit = hyp_decline(t_forecast, qi, D, b)
                        fig_dca.add_trace(go.Scatter(x=date_forecast, y=q_fit, mode="lines", name="Hyperbolic",
                                                     line=dict(width=3, dash="longdash", color="#9467bd"), yaxis="y1"))
                        cum_fit = np.cumsum(q_fit)
                        cumu_traces.append(go.Scatter(x=date_forecast, y=cum_fit, mode="lines", name="Cumulative Hyperbolic",
                                                     line=dict(width=2, dash="longdash", color="#9467bd"), yaxis="y2"))
                        # EUR for hyperbolic: EUR = (qi / ((1-b) * D)) * [1 - (1 + b*D*t_end)^{(b-1)/b} ] for b != 1
                        t_end = t_forecast[-1]
                        if D > 0 and b > 0 and b != 1:
                            EUR = (qi / ((1-b) * D)) * (1 - (1 + b*D*t_end) ** ((b-1)/b))
                        elif D > 0 and b == 1:
                            EUR = qi / D * np.log(1 + D * t_end)
                        else:
                            EUR = np.nan
                        results.append({"Method": "Hyperbolic", "qi": qi, "D": D, "b": b, "EUR": EUR})
                    except Exception:
                        results.append({"Method": "Hyperbolic", "qi": np.nan, "D": np.nan, "b": np.nan, "EUR": np.nan})
            # Add cumulative production traces to the figure
            for trace in cumu_traces:
                fig_dca.add_trace(trace)
            fig_dca.update_layout(
                title=dict(text="<b>Decline Curve Analysis</b>", font=dict(size=22)),
                xaxis_title=dict(text="<b>Date</b>", font=dict(size=16)),
                yaxis=dict(title="<b>Rate</b>", side="left", titlefont=dict(size=16), showgrid=True, gridwidth=1, gridcolor='#e5e5e5'),
                yaxis2=dict(title="<b>Cumulative Production</b>", overlaying="y", side="right", titlefont=dict(size=16), showgrid=False),
                plot_bgcolor="#f9f9f9",
                legend=dict(font=dict(size=13), bgcolor='rgba(255,255,255,0.7)', bordercolor='#ccc', borderwidth=1, x=0.01, y=0.99),
                margin=dict(l=60, r=30, t=60, b=60),
                hovermode="x unified"
            )
            fig_dca.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
            st.plotly_chart(fig_dca, use_container_width=True)
            # Show table of results
            st.subheader("Calculated Decline Parameters")
            df_results = pd.DataFrame(results)
            df_results = df_results.rename(columns={"qi": "Initial Rate (qi)", "D": "Decline Rate (D)", "b": "b (hyperbolic)", "EUR": "Estimated Ultimate Recovery (EUR)"})
            st.dataframe(df_results)

# --------------------------
# PTA TAB
# --------------------------
with tab_pta:
    st.header("Pressure Transient Analysis (Horner & Derivative)")

    pta_file = st.file_uploader("Upload PTA buildup data (Excel/CSV) with columns 't(hr)' and 'pws(psig)'",
                                type=["xlsx","csv"], key="pta_uploader")

    if pta_file is None:
        st.info("Upload a PTA file or sample data to begin.")
    else:
        # load
        if pta_file.name.endswith(".xlsx"):
            df_pta = pd.read_excel(pta_file)
        else:
            df_pta = pd.read_csv(pta_file)

        st.subheader("Raw PTA Data")
        st.dataframe(df_pta)

        expected = {"t(hr)","pws(psig)"}
        if not expected.issubset(df_pta.columns):
            st.error(f"File must contain columns: {expected}")
        else:
            # clean and sort
            df_pta = df_pta.sort_values("t(hr)").drop_duplicates(subset=["t(hr)"])
            t_all = df_pta["t(hr)"].to_numpy(dtype=float)
            pws_all = df_pta["pws(psig)"].to_numpy(dtype=float)

            # user inputs for PTA
            st.subheader("PTA Inputs (manual)")
            col1, col2 = st.columns(2)
            with col1:
                tp = st.number_input("Production time tp (hr)", value=310.0, key="tp_pta")
                Qo_pta = st.number_input("Test oil rate Qo (STB/day)", value=4900.0, key="Qo_pta")
                Bo_pta = st.number_input("Bo (bbl/STB)", value=1.55, key="Bo_pta")
                mu_pta = st.number_input("Viscosity μo (cp)", value=0.20, key="mu_pta")
                h_pta = st.number_input("Net thickness h (ft)", value=482.0, key="h_pta")
            with col2:
                phi_pta = st.number_input("Porosity φ", value=0.09, key="phi_pta")
                ct_pta = st.number_input("Total compressibility ct (psi^-1)", value=22.6e-6, format="%.6e", key="ct_pta")
                rw_pta = st.number_input("Wellbore radius rw (ft)", value=0.354, key="rw_pta")
                re_pta = st.number_input("Drainage radius re (ft)", value=2640.0, key="re_pta")
                pwf0_pta = st.number_input("pwf(t=0) (psi)", value=float(pws_all[0]) if len(pws_all)>0 else 2761.0, key="pwf0_pta")

            # Remove t==0 for horner/log operations but keep original for display
            mask_pos = t_all > 0
            if not np.any(mask_pos):
                st.error("No positive shut-in times in data (t>0 required).")
            else:
                t = t_all[mask_pos]
                pws = pws_all[mask_pos]
                tp_t = tp + t
                horner = tp_t / t
                log_horner = np.log10(horner)

                # Horner plot + regression
                st.subheader("Horner Plot (interactive)")
                portion = st.slider("Select % of last points used for slope regression", min_value=10, max_value=90, value=50, step=5, key="portion_pta")
                start_idx = int(len(log_horner) * (1 - portion/100.0))
                start_idx = max(0, start_idx)
                x_reg = log_horner[start_idx:]
                y_reg = pws[start_idx:]

                # Fit on selected portion to get slope (psi per log cycle)
                coeffs = np.polyfit(x_reg, y_reg, 1)
                slope_fit = coeffs[0]
                m_calc = -slope_fit    # psi per log cycle (convention)
                intercept_fit = coeffs[1]
                st.write(f"Calculated slope m = {m_calc:.4f} psi per log cycle (from last {portion}% of points)")

                # allow user to choose reference point (closest to 1 hr or pick index)
                choose_ref = st.radio("Reference point for straight line:", options=["Closest to 1 hr", "Choose index from data"], index=0)
                if choose_ref == "Closest to 1 hr":
                    ref_time = st.number_input("Reference time (hr) used if 'Closest to 1 hr'", value=1.0, key="reftime_pta")
                    ref_idx = np.argmin(np.abs(t - ref_time))
                else:
                    idx_val = st.slider("Choose data index for reference (index in sorted positive-time data)", min_value=0, max_value=len(t)-1, value=min(0, len(t)-1), key="ref_idx_pta")
                    ref_idx = idx_val

                x_ref_log = log_horner[ref_idx]
                y_ref = pws[ref_idx]

                # build straight line in log space and convert back to Horner axis for plotting (x axis is horner)
                xline_log = np.linspace(np.min(log_horner), np.max(log_horner), 200)
                yline = y_ref + m_calc * (xline_log - x_ref_log)
                xline_horner = 10**xline_log

                # Interactive plotly Horner plot (x-axis = horner, log scale)
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=horner, y=pws, mode='markers+lines', name='Horner data',
                                           marker=dict(size=8, color='#1f77b4'),
                                           line=dict(width=2, color='#1f77b4')))
                fig_h.add_trace(go.Scatter(x=xline_horner, y=yline, mode='lines', name=f'Straight line (m={m_calc:.3f})',
                                           line=dict(dash='dash', color='#d62728', width=3)))
                fig_h.update_layout(
                    title=dict(text="<b>Horner Plot (x=(tp+t)/t, log scale)</b>", font=dict(size=22)),
                    xaxis_type="log",
                    xaxis_title=dict(text="<b>Horner time (tp+t)/t</b>", font=dict(size=18)),
                    yaxis_title=dict(text="<b>Pressure (psi)</b>", font=dict(size=18)),
                    hovermode="x unified",
                    plot_bgcolor="#f9f9f9",
                    legend=dict(font=dict(size=14), bgcolor='rgba(255,255,255,0.7)', bordercolor='#ccc', borderwidth=1, x=0.01, y=0.99),
                    margin=dict(l=60, r=30, t=60, b=60)
                )
                fig_h.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
                fig_h.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
                st.plotly_chart(fig_h, use_container_width=True)

                # show regression portion highlighted
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=10**log_horner, y=pws, mode='markers+lines', name='All data',
                                             marker=dict(size=7, color='#1f77b4'),
                                             line=dict(width=2, color='#1f77b4')))
                fig_reg.add_trace(go.Scatter(x=10**x_reg, y=y_reg, mode='markers', name=f'Regression portion (last {portion}%)',
                                             marker=dict(size=12, color='#ff7f0e', symbol='diamond')))
                fig_reg.update_layout(
                    title=dict(text="<b>Regression Portion Highlighted</b>", font=dict(size=20)),
                    xaxis_type="log",
                    xaxis_title=dict(text="<b>Horner time (tp+t)/t</b>", font=dict(size=16)),
                    yaxis_title=dict(text="<b>Pressure (psi)</b>", font=dict(size=16)),
                    plot_bgcolor="#f9f9f9",
                    legend=dict(font=dict(size=13), bgcolor='rgba(255,255,255,0.7)', bordercolor='#ccc', borderwidth=1, x=0.01, y=0.99),
                    margin=dict(l=60, r=30, t=50, b=50)
                )
                fig_reg.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
                fig_reg.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
                st.plotly_chart(fig_reg, use_container_width=True)

                # ============================
                # Derivative analysis (three-point weighted) using log10(t) as x
                # ============================
                st.subheader("Derivative Analysis (three-point weighted)")

                logt = np.log10(t)
                N = len(t)
                dpdlogt = np.full(N, np.nan, dtype=float)
                for i in range(1, N-1):
                    dp1 = pws[i] - pws[i-1]
                    dx1 = logt[i] - logt[i-1]
                    dp2 = pws[i+1] - pws[i]
                    dx2 = logt[i+1] - logt[i]
                    if dx1 == 0 or dx2 == 0:
                        dpdlogt[i] = np.nan
                    else:
                        dpdlogt[i] = ((dp1/dx1)*dx2 + (dp2/dx2)*dx1) / (dx1 + dx2)

                # interactive derivative plot
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(x=t, y=pws, mode='lines+markers', name='Pressure (psi)',
                                           marker=dict(size=7, color='#1f77b4'),
                                           line=dict(width=2, color='#1f77b4'),
                                           yaxis='y1'))
                fig_d.add_trace(go.Scatter(x=t, y=dpdlogt, mode='lines+markers', name='dp/dlog(t)',
                                           marker=dict(size=7, color='#2ca02c', symbol='cross'),
                                           line=dict(width=2, color='#2ca02c', dash='dot'),
                                           yaxis='y2'))
                fig_d.update_layout(
                    title=dict(text="<b>Pressure and Derivative vs Time (log x-axis)</b>", font=dict(size=20)),
                    xaxis=dict(title='<b>Time (hr)</b>', type='log', titlefont=dict(size=16)),
                    yaxis=dict(title='<b>Pressure (psi)</b>', side='left', titlefont=dict(size=16)),
                    yaxis2=dict(title='<b>Derivative (psi per log cycle)</b>', overlaying='y', side='right', titlefont=dict(size=16)),
                    hovermode='x unified',
                    plot_bgcolor="#f9f9f9",
                    legend=dict(font=dict(size=13), bgcolor='rgba(255,255,255,0.7)', bordercolor='#ccc', borderwidth=1, x=0.01, y=0.99),
                    margin=dict(l=60, r=30, t=50, b=50)
                )
                fig_d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
                fig_d.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
                st.plotly_chart(fig_d, use_container_width=True)

                # ============================
                # Compute permeability, skin, and p_skin using Horner results
                # ============================
                st.subheader("Derived Reservoir Parameters from Horner analysis")
                # Permeability using formula k = (162.6 * Qo * Bo * mu) / (m * h)
                # Use Qo_pta (test rate), Bo_pta, mu_pta, h_pta
                if m_calc == 0:
                    st.warning("Calculated slope m is zero → cannot compute permeability.")
                    k_calc = np.nan
                else:
                    k_calc = (162.6 * Qo_pta * Bo_pta * mu_pta) / (m_calc * h_pta)

                st.write(f"Calculated permeability k = {k_calc:.3f} md (using Qo, Bo, μ, h and slope m)")

                # p1hr - pick p at t closest to 1 hr or interpolate
                t_target = 1.0
                if np.any(np.isclose(t, t_target, atol=1e-6)):
                    p1hr = pws[np.argmin(np.abs(t - t_target))]
                else:
                    # interpolation in log time space (or linear)
                    try:
                        p1hr = np.interp(t_target, t, pws)
                    except Exception:
                        p1hr = pws[0]

                # skin factor s using earlier equation used in conversation:
                # s = 1.151 * ( (p1hr - pwf0) / m - log10( k/(phi * mu * ct * rw^2) ) + 3.23 )
                if m_calc == 0 or np.isnan(k_calc):
                    s_calc = np.nan
                else:
                    term = np.log10((k_calc) / (phi_pta * mu_pta * ct_pta * rw_pta**2))
                    s_calc = 1.151 * (((p1hr - pwf0_pta) / m_calc) - term + 3.23)

                st.write(f"Estimated skin factor s = {s_calc:.3f}")

                # additional pressure drop due to skin
                if np.isnan(s_calc) or np.isnan(m_calc):
                    dp_skin = np.nan
                else:
                    dp_skin = 0.87 * m_calc * s_calc
                st.write(f"Additional pressure drop due to skin Δp_skin = {dp_skin:.2f} psi")

# --------------------------
# NODAL TAB
# --------------------------
with tab_nodal:
    st.header("Nodal Analysis (IPR & TPR)")

    # Sidebar inputs
    st.sidebar.header("Reservoir & Fluid Inputs (IPR)")
    pr = st.sidebar.number_input("Average reservoir pressure p_r (psi)", value=3320.0)
    k = st.sidebar.number_input("Permeability k (md)", value=12.8)
    h = st.sidebar.number_input("Net thickness h (ft)", value=482.0)
    mu = st.sidebar.number_input("Viscosity μo (cp)", value=0.20)
    Bo = st.sidebar.number_input("Bo (bbl/STB)", value=1.55)
    re = st.sidebar.number_input("Drainage radius re (ft)", value=2640.0)
    rw = st.sidebar.number_input("Wellbore radius rw (ft)", value=0.354)
    s = st.sidebar.number_input("Skin factor s", value=0.0)

    st.sidebar.header("Tubing & Flow Inputs (TPR)")
    pwh = st.sidebar.number_input("Wellhead pressure pwh (psi)", value=100.0)
    TVD = st.sidebar.number_input("True Vertical depth TVD (ft)", value=10476.0)
    MD = st.sidebar.number_input("Measured depth MD (ft)", value=10476.0)
    ID_inch = st.sidebar.number_input("Tubing inner diameter (in)", value=2.441)
    API = st.number_input("Oil API Gravity", min_value=10.0, max_value=60.0, value=35.0)
    mu_tpr = st.sidebar.number_input("Viscosity for TPR (cp)", value=0.20)
    roughness_in = st.sidebar.number_input("Absolute roughness (in)", value=0.0005)
    roughness_ft = roughness_in / 12.0
    fric_method = st.sidebar.selectbox("Friction factor method", ["colebrook","haaland","chen"], index=0)

    
    SG = 141.5 / (API + 131.5)
    rho_lb_ft3 = SG * 62.4  # lb/ft³
    
    # Productivity index J
    denom = np.log(re/rw) - 0.75 + s
    if denom == 0:
        J = np.nan
        st.error("Denominator in J equation is zero!")
    else:
        J = (0.00708 * k * h) / (mu * Bo * denom)
    st.write(f"Productivity index J = {J:.6f} STB/day/psi")

    # IPR curve
    # Input: reservoir pressure pr, test rate q_test, pwf_test
    ipr_method = st.radio("Select IPR Method", ["Darcy", "Vogel"])

    pwf_vals = np.arange(0, pr, 500)

    if ipr_method == "Darcy":
        # Calculate J (Productivity Index)
        q_ipr = J * (pr - pwf_vals)
        AOF = J * pr
    elif ipr_method == "Vogel":
        # Estimate q_max (AOF) from test point
        Q_max = (J * pr) / 1.8
        q_ipr= Q_max * (1 - 0.2 * (pwf_vals/ pr) - 0.8 * (pwf_vals/ pr)**2)
        AOF = Q_max
    st.write(f"Absolute Open Flow (AOF) = {AOF:.2f} STB/day")

    # TPR curve
    v_range = (4 * q_ipr * 5.615) / (np.pi * ((ID_inch/12)**2) * 86400)  # velocity in ft/s
    q_range = q_ipr
    hydrostatic_psi = (rho_lb_ft3 * TVD) / 144  # convert lbf/ft2 to psi
    pwf_tpr_list = []

    for qi, vi in zip(q_range, v_range):
        dp_per_ft = dp_friction_per_ft(qi, vi, ID_inch, roughness_ft, rho_lb_ft3, mu_tpr, MD)
        dp_total_psi = (dp_per_ft * MD) / 144  # convert lbf/ft2 to psi
        pwf_tpr = pwh + hydrostatic_psi + dp_total_psi
        pwf_tpr_list.append(pwf_tpr)
    pwf_tpr_arr = np.array(pwf_tpr_list)

    # Nodal plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_ipr, y=pwf_vals, mode="lines", name="IPR",
                             line=dict(width=3, color="#1f77b4")))
    fig.add_trace(go.Scatter(x=q_range, y=pwf_tpr_arr, mode="lines", name=f"TPR ({fric_method})",
                             line=dict(width=3, color="#d62728", dash="dash")))
    fig.update_layout(
        title=dict(text="<b>IPR vs TPR</b>", font=dict(size=24)),
        xaxis_title=dict(text="<b>q (STB/day)</b>", font=dict(size=18)),
        yaxis_title=dict(text="<b>pwf (psi)</b>", font=dict(size=18)),
        yaxis=dict(range=[0, pr+50]),
        plot_bgcolor="#f9f9f9",
        legend=dict(font=dict(size=15), bgcolor='rgba(255,255,255,0.7)', bordercolor='#ccc', borderwidth=1, x=0.01, y=0.99),
        margin=dict(l=60, r=30, t=60, b=60),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e5e5')
    st.plotly_chart(fig, use_container_width=True)

    # Table of values
    st.subheader("Calculated IPR and TPR Data")
    df_out = pd.DataFrame({
        "pwf_IPR (psi)": pwf_vals,
        "q_IPR (STB/day)": q_ipr
    })
    df_tpr = pd.DataFrame({
        "q_TPR (STB/day)": q_range,
        "pwf_TPR (psi)": pwf_tpr_arr,
        "Velocity":v_range,

    })
    st.write("**IPR Table**")
    st.dataframe(df_out)
    st.write("**TPR Table**")
    st.dataframe(df_tpr)

    # Direct data-based intersection: find where pwf_IPR and pwf_TPR are closest
    # Interpolate TPR to IPR q values (or vice versa)
    from scipy.interpolate import interp1d
    # Remove any NaNs for interpolation
    mask = (~np.isnan(q_range)) & (~np.isnan(pwf_tpr_arr))
    interp_tpr = interp1d(q_range[mask], pwf_tpr_arr[mask], bounds_error=False, fill_value=np.nan)
    pwf_tpr_on_ipr = interp_tpr(q_ipr)
    diff = np.abs(pwf_vals - pwf_tpr_on_ipr)
    idx_star = np.nanargmin(diff)
    q_star = q_ipr[idx_star]
    pwf_star = pwf_vals[idx_star]
    st.success(f"Operating point (IPR=TPR, data-based):\nq* = {q_star:.2f} STB/day\npwf* = {pwf_star:.2f} psi")
    fig.add_trace(go.Scatter(x=[q_star], y=[pwf_star], mode="markers", name="Operating Point (data)",
                             marker=dict(size=18, color="black", symbol="circle")))
    st.plotly_chart(fig, use_container_width=True)