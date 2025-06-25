import math
import os
import matplotlib
matplotlib.use('Agg') # Important for running matplotlib in a web server environment
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, url_for
import time # <--- ADDED IMPORT FOR TIME

app = Flask(__name__)

plots_dir_setup = os.path.join(app.static_folder, 'plots')
if not os.path.exists(plots_dir_setup):
    os.makedirs(plots_dir_setup)

density = 1000
vapor_pressure_bar = 1.0
viscosity = 0.001
g = 9.81
P_atm_bar = 1.01325

def convert_flowrate_to_m3s(flowrate_val, unit):
    if unit == "m3/s": return flowrate_val, flowrate_val * 3600
    elif unit == "m3/h": return flowrate_val / 3600, flowrate_val
    elif unit == "l/s": return flowrate_val / 1000, flowrate_val * 3.6
    return 0, 0

def convert_diameter_to_m(diameter_val, unit):
    if unit == "m": return diameter_val, diameter_val * 1000, diameter_val / 0.0254
    elif unit == "mm": return diameter_val / 1000, diameter_val, (diameter_val / 1000) / 0.0254
    elif unit == "in": return diameter_val * 0.0254, (diameter_val * 0.0254) * 1000, diameter_val
    return 0, 0, 0

def calculate_velocity(flowrate_m3s, diameter_m):
    if diameter_m == 0: return 0, 0
    area = math.pi * (diameter_m / 2) ** 2
    return area, flowrate_m3s / area if area != 0 else 0

def calculate_reynolds_number(velocity, diameter_m, viscosity_pas, density_kgm3):
    if density_kgm3 == 0 or diameter_m == 0: return 0, 0
    kin_visc = viscosity_pas / density_kgm3 if density_kgm3 != 0 else 0
    if kin_visc == 0: return 0, 0
    Re = velocity * diameter_m / kin_visc
    return round(Re, 2), round(kin_visc, 8)

def colebrook_white(diameter_m, roughness_m, Re):
    if Re == 0 or diameter_m == 0: return 0.02
    if Re < 2300: return 64 / Re if Re > 0 else 0.02
    lam = 0.02
    for _ in range(100):
        if lam == 0: lam = 1e-8
        term1_denominator = (3.7 * diameter_m)
        term1 = roughness_m / term1_denominator if term1_denominator != 0 else 0
        sqrt_lam = math.sqrt(lam)
        term2_denominator = (Re * sqrt_lam)
        term2 = 2.51 / term2_denominator if term2_denominator != 0 else 0
        log_arg = term1 + term2
        if log_arg <= 0: return lam
        try: right_hand_side = -2.0 * math.log10(log_arg)
        except ValueError: return lam
        if right_hand_side == 0: return lam
        lam_new = (1 / right_hand_side) ** 2
        if abs(lam - lam_new) < 1e-7: return lam_new
        lam = lam_new
    return lam

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    plot_urls = {}
    error_message = None
    form_data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            flowrate_val = float(form_data['flowrate'])
            flowrate_unit = form_data['flowrate_unit']
            diameter_val = float(form_data['diameter'])
            diameter_unit = form_data['diameter_unit']
            suction_elev = float(form_data['suction_elev'])
            discharge_elev = float(form_data['discharge_elev'])
            pipe_length = float(form_data['pipe_length'])
            friction_method = form_data['friction_method']
            hazen_williams_c = float(form_data.get('hazen_williams_c', 130))
            pipe_roughness_mm = float(form_data.get('pipe_roughness_mm', 0.05))
            pump_efficiency_percent = float(form_data['pump_efficiency'])
            operating_hours = float(form_data['operating_hours'])
            unit_cost_electricity = float(form_data['unit_cost_electricity'])
            reservoir_liters = float(form_data['reservoir_liters'])

            if pipe_length <= 0: raise ValueError("Pipe length must be greater than zero.")
            if not 0 < pump_efficiency_percent <= 100: raise ValueError("Pump efficiency must be 0-100.")

            flowrate_m3s, flowrate_m3h = convert_flowrate_to_m3s(flowrate_val, flowrate_unit)
            diameter_m, diameter_mm, diameter_in = convert_diameter_to_m(diameter_val, diameter_unit)
            area, velocity = calculate_velocity(flowrate_m3s, diameter_m)
            Re, kin_viscosity = calculate_reynolds_number(velocity, diameter_m, viscosity, density)
            flow_type = "Laminar" if Re < 2300 else "Transitional" if Re < 4000 else "Turbulent"

            Hs = round(discharge_elev - suction_elev, 3) 
            Hv = round((velocity ** 2) / (2 * g), 3) if g != 0 else 0 

            Ps_bar = (suction_elev - Hv) * density * g / 100000 if (density * g != 0) else 0
            Pd_bar = (discharge_elev - Hv) * density * g / 100000 if (density * g != 0) else 0
            Ps_bar_val = Ps_bar if Ps_bar is not None else 0
            Pd_bar_val = Pd_bar if Pd_bar is not None else 0
            dP_bar = Pd_bar_val - Ps_bar_val
            Hp_m = (dP_bar * 100000) / (density * g) if (density * g != 0) else 0 

            Hf = Sf = lambda_val = Re_star = None
            turbulence_type = ""

            if friction_method == "hazen_williams":
                if hazen_williams_c == 0 or diameter_m == 0: Hf = float('inf') if flowrate_m3s > 0 else 0
                else: Hf = (10.67*pipe_length*(flowrate_m3s**1.852))/((hazen_williams_c**1.852)*(diameter_m**4.8704))
                Sf = Hf / pipe_length if pipe_length != 0 else 0
            elif friction_method == "colebrook_white":
                roughness_m = pipe_roughness_mm / 1000
                lambda_val = colebrook_white(diameter_m, roughness_m, Re)
                Hf = (lambda_val*pipe_length*velocity**2)/(2*g*diameter_m) if (g!=0 and diameter_m!=0) else 0
                Sf = Hf / pipe_length if pipe_length != 0 else 0
                if Re > 0 and diameter_m > 0 and lambda_val is not None and lambda_val > 0:
                    Re_star = Re*(roughness_m/diameter_m)*math.sqrt(lambda_val/8) if diameter_m!=0 else 0
                    turbulence_type = ("Smooth" if Re_star<4 else "Transitional" if Re_star<=60 else "Rough")+" turbulent"
                else: Re_star = 0; turbulence_type = "N/A"
            
            H = Hs + Hv + (Hf if Hf is not None else 0) + Hp_m

            abs_pressure_at_suction_source_m = (P_atm_bar * 100000 ) / (density*g) if (density*g!=0) else 0
            vapor_pressure_head_m = (vapor_pressure_bar * 100000) / (density*g) if (density*g!=0) else 0
            Hf_for_npsha = Hf if Hf is not None else 0
            NPSH_available = abs_pressure_at_suction_source_m + suction_elev - vapor_pressure_head_m - (0.3*Hf_for_npsha) - Hv

            hydraulic_power_kw = (flowrate_m3s*density*g*H)/1000 if (density*g!=0) else 0
            efficiency_pump = pump_efficiency_percent / 100
            if efficiency_pump == 0: raise ValueError("Pump efficiency cannot be zero.")
            pump_shaft_power_kw = hydraulic_power_kw / efficiency_pump 
            motor_efficiency_assumed = 0.85
            actual_electrical_power_kw = pump_shaft_power_kw/motor_efficiency_assumed if motor_efficiency_assumed!=0 else float('inf')
            overall_efficiency = efficiency_pump * motor_efficiency_assumed
            energy_consumed_kwh = actual_electrical_power_kw * operating_hours
            operating_cost = energy_consumed_kwh * unit_cost_electricity
            time_to_fill_hr = 0
            if flowrate_m3h > 0:
                reservoir_m3 = reservoir_liters / 1000
                time_to_fill_hr = reservoir_m3 / flowrate_m3h
            time_to_fill_min = time_to_fill_hr * 60
            time_to_fill_sec = time_to_fill_hr * 3600

            results = {
                "flowrate_m3h":f"{flowrate_m3h:.2f}", "flowrate_m3s":f"{flowrate_m3s:.6f}",
                "diameter_in":f"{diameter_in:.4f}", "diameter_mm":f"{diameter_mm:.2f}", "diameter_m":f"{diameter_m:.4f}",
                "pipe_area_m2":f"{area:.6f}", "velocity_ms":f"{velocity:.2f}",
                "reynolds_number":f"{Re:.0f}" if Re is not None else "N/A", "flow_type":flow_type,
                "kin_viscosity_m2s":f"{kin_viscosity:.8f}",
                "static_head_m":f"{Hs:.2f}", "velocity_head_m":f"{Hv:.2f}",
                "ps_bar": f"{Ps_bar:.2f}" if Ps_bar is not None else "N/A",
                "pd_bar": f"{Pd_bar:.2f}" if Pd_bar is not None else "N/A",
                "dp_bar": f"{dP_bar:.2f}" if dP_bar is not None else "N/A",
                "hp_m": f"{Hp_m:.2f}" if Hp_m is not None else "N/A",
                "friction_factor_lambda":f"{lambda_val:.4f}" if lambda_val is not None else "N/A",
                "friction_head_m":f"{Hf:.2f}" if Hf is not None else "N/A", 
                "friction_slope":f"{Sf:.5f}" if Sf is not None else "N/A",
                "total_head_m":f"{H:.2f}", "npsh_available_m":f"{NPSH_available:.2f}",
                "reynolds_roughness_Re_star":f"{Re_star:.2f}" if Re_star is not None and Re_star!=0 else "N/A",
                "turbulence_type":turbulence_type if turbulence_type else "N/A",
                "hydraulic_power_kw":f"{hydraulic_power_kw:.3f}",
                "pump_efficiency_percent":f"{efficiency_pump*100:.2f}",
                "pump_shaft_power_kw":f"{pump_shaft_power_kw:.3f}", 
                "motor_efficiency_assumed_percent":f"{motor_efficiency_assumed*100:.2f}",
                "overall_efficiency_percent":f"{overall_efficiency*100:.2f}",
                "actual_electrical_power_kw":f"{actual_electrical_power_kw:.3f}",
                "energy_consumed_kwh":f"{energy_consumed_kwh:.3f}", "power_cost":f"{operating_cost:.0f}",
                "reservoir_liters":f"{reservoir_liters:.0f}",
                "fill_time_hr":f"{time_to_fill_hr:.2f}", "fill_time_min":f"{time_to_fill_min:.2f}", "fill_time_sec":f"{time_to_fill_sec:.2f}",
            }

            # --- PLOTTING ---
            plot_save_dir = os.path.join(app.static_folder, 'plots')
            cache_bust_val = int(time.time()) # <--- CORRECTED CACHE BUSTING

            flowrate_range_plot_m3h = np.linspace(0.01, max(0.01, flowrate_m3h) * 1.5, 100)

            # 1. System Curve & Pump Curve
            plt.figure(figsize=(8, 5))
            system_head_values = []
            for q_plot_m3h in flowrate_range_plot_m3h:
                q_plot_m3s = q_plot_m3h / 3600
                _, vel_plot = calculate_velocity(q_plot_m3s, diameter_m)
                re_plot, _ = calculate_reynolds_number(vel_plot, diameter_m, viscosity, density)
                hf_plot = 0
                if friction_method == "hazen_williams":
                    if hazen_williams_c==0 or diameter_m==0: hf_plot=float('inf') if q_plot_m3s>0 else 0
                    else: hf_plot=(10.67*pipe_length*(q_plot_m3s**1.852))/((hazen_williams_c**1.852)*(diameter_m**4.8704))
                elif friction_method == "colebrook_white":
                    lambda_plot = colebrook_white(diameter_m, pipe_roughness_mm/1000, re_plot)
                    hf_plot = (lambda_plot*pipe_length*vel_plot**2)/(2*g*diameter_m) if (g>0 and diameter_m>0) else 0
                hv_plot = (vel_plot**2 / (2*g) if g>0 else 0)
                # Recalculate Ps_plot, Pd_plot, Hp_m_plot for system curve consistency
                Ps_bar_plot = (suction_elev - hv_plot) * density * g / 100000 if (density * g != 0) else 0
                Pd_bar_plot = (discharge_elev - hv_plot) * density * g / 100000 if (density * g != 0) else 0
                dP_bar_plot = Pd_bar_plot - Ps_bar_plot
                Hp_m_plot = (dP_bar_plot * 100000) / (density * g) if (density * g != 0) else 0

                h_plot = Hs + hv_plot + (hf_plot if hf_plot is not None else 0) + Hp_m_plot 
                system_head_values.append(h_plot)
            plt.plot(flowrate_range_plot_m3h, system_head_values, label="System Head Curve (Hsys)")
            H_op_point = H if H is not None else 0
            flowrate_op_point_m3h = flowrate_m3h if flowrate_m3h is not None else 0.01
            H0_pump = H_op_point*1.2; a_pump=(H0_pump-H_op_point)/(flowrate_op_point_m3h**2) if flowrate_op_point_m3h>0 else 0
            pump_head_curve = np.maximum(0, H0_pump - a_pump * (flowrate_range_plot_m3h**2))
            plt.plot(flowrate_range_plot_m3h, pump_head_curve, label="Example Pump Curve (Hp)", linestyle='--')
            if H is not None and flowrate_m3h is not None: plt.scatter([flowrate_m3h], [H], color='red', zorder=5, label="Op. Point")
            plt.xlabel("Flowrate (m³/h)"); plt.ylabel("Head (m)"); plt.title("System & Pump Curves")
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.ylim(bottom=0)
            plt.savefig(os.path.join(plot_save_dir, 'system_curve.png')); plt.close()
            plot_urls['system_curve'] = url_for('static', filename='plots/system_curve.png', t=cache_bust_val)

            # 2. Pump Efficiency vs Flowrate
            plt.figure(figsize=(8, 5))
            bep_flowrate_m3h = flowrate_m3h if flowrate_m3h is not None and flowrate_m3h > 0 else 1 
            eff_pump_val = efficiency_pump if efficiency_pump is not None else 0.75
            eff_curve_values = eff_pump_val * np.exp(-((flowrate_range_plot_m3h - bep_flowrate_m3h)**2) / (2*(0.4*bep_flowrate_m3h+1e-6)**2))
            eff_curve_values = np.clip(eff_curve_values * 100, 0, 100)
            plt.plot(flowrate_range_plot_m3h, eff_curve_values)
            if efficiency_pump is not None and flowrate_m3h is not None: plt.scatter([flowrate_m3h], [efficiency_pump*100], color='red', zorder=5, label="Op. Eff.")
            plt.xlabel("Flowrate (m³/h)"); plt.ylabel("Pump Efficiency (%)"); plt.title("Assumed Pump Efficiency Curve")
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.ylim(0, 105)
            plt.savefig(os.path.join(plot_save_dir, 'efficiency_vs_flowrate.png')); plt.close()
            plot_urls['efficiency'] = url_for('static', filename='plots/efficiency_vs_flowrate.png', t=cache_bust_val)
            
            # 3. NPSHa vs Flowrate 
            plt.figure(figsize=(8, 5))
            npsha_values_plot = []
            for q_plot_m3h in flowrate_range_plot_m3h:
                q_plot_m3s = q_plot_m3h / 3600; _, vel_plot = calculate_velocity(q_plot_m3s, diameter_m)
                re_plot, _ = calculate_reynolds_number(vel_plot, diameter_m, viscosity, density)
                hf_total_plot = 0
                if friction_method=="hazen_williams":
                     if hazen_williams_c==0 or diameter_m==0: hf_total_plot=float('inf') if q_plot_m3s>0 else 0
                     else: hf_total_plot=(10.67*pipe_length*(q_plot_m3s**1.852))/((hazen_williams_c**1.852)*(diameter_m**4.8704))
                elif friction_method=="colebrook_white":
                    lambda_plot_npsh = colebrook_white(diameter_m, pipe_roughness_mm/1000, re_plot)
                    hf_total_plot = (lambda_plot_npsh*pipe_length*vel_plot**2)/(2*g*diameter_m) if (g>0 and diameter_m>0) else 0
                hf_suction_plot_approx = 0.3*hf_total_plot; hv_plot=(vel_plot**2/(2*g) if g>0 else 0)
                # For NPSHa plot, abs_pressure_at_suction_source_m is based on P_atm, not varying Ps_bar_plot
                npsha_val = abs_pressure_at_suction_source_m + suction_elev - vapor_pressure_head_m - hf_suction_plot_approx - hv_plot
                npsha_values_plot.append(max(0, npsha_val))
            plt.plot(flowrate_range_plot_m3h, npsha_values_plot, label="NPSH Avail. (m)")
            npsha_op_point = NPSH_available if NPSH_available is not None else 0
            flowrate_op_point_m3h_npsh = flowrate_m3h if flowrate_m3h is not None and flowrate_m3h > 0 else 0.01
            npshr_at_op_point = max(0.5, npsha_op_point - 1) 
            k_npshr = npshr_at_op_point / (flowrate_op_point_m3h_npsh**2) if flowrate_op_point_m3h_npsh > 0 else 0
            npshr_curve = np.maximum(0.1, k_npshr * (flowrate_range_plot_m3h**2))
            plt.plot(flowrate_range_plot_m3h, npshr_curve, label="NPSH Req. (example)", linestyle='--')
            if NPSH_available is not None and flowrate_m3h is not None: plt.scatter([flowrate_m3h], [NPSH_available], color='red', zorder=5, label="Op. NPSHa")
            plt.xlabel("Flowrate (m³/h)"); plt.ylabel("NPSH (m)"); plt.title("NPSH Avail. vs. Req. (Example)");
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.ylim(bottom=0)
            plt.savefig(os.path.join(plot_save_dir, 'npsh_vs_flowrate.png')); plt.close()
            plot_urls['npsh'] = url_for('static', filename='plots/npsh_vs_flowrate.png', t=cache_bust_val)

            # 4. Head Loss Breakdown Bar Chart
            plt.figure(figsize=(7, 5))
            head_components = ['Static (Hs)', 'Velocity (Hv)', 'Friction (Hf)', 'Pressure (Hp_m)']
            Hf_bar = Hf if Hf is not None else 0
            Hp_m_bar = Hp_m if Hp_m is not None else 0
            head_values_bar = [Hs, Hv, Hf_bar, Hp_m_bar]
            plt.bar(head_components, head_values_bar, color=['cornflowerblue', 'gold', 'salmon', 'lightgreen'])
            plt.ylabel("Head (m)"); plt.title("Total Head Components (H)"); plt.xticks(rotation=10, ha="right"); plt.tight_layout()
            plt.savefig(os.path.join(plot_save_dir, 'head_loss_breakdown.png')); plt.close()
            plot_urls['head_loss'] = url_for('static', filename='plots/head_loss_breakdown.png', t=cache_bust_val)
            
            # 5. Power Breakdown Bar Chart
            plt.figure(figsize=(7, 5))
            power_labels=['Hydraulic', 'Pump Shaft', 'Electrical']
            power_values_bar=[hydraulic_power_kw, pump_shaft_power_kw, actual_electrical_power_kw]
            plt.bar(power_labels, power_values_bar, color=['skyblue', 'lightcoral', 'mediumseagreen'])
            plt.ylabel("Power (kW)"); plt.title("Power Flow"); plt.tight_layout()
            plt.savefig(os.path.join(plot_save_dir, 'power_comparison.png')); plt.close()
            plot_urls['power_comp'] = url_for('static', filename='plots/power_comparison.png', t=cache_bust_val)

        except ValueError as e: 
            error_message = f"Input Error: {e}"
            results = None
        except ZeroDivisionError as e: 
            error_message = f"Calc Error: Division by zero. Check inputs. ({e})"
            results = None
        except Exception as e: 
            error_message = f"Unexpected error: {e}"
            results = None
            import traceback
            traceback.print_exc()
        
        if error_message and results is None: 
            plot_urls = {}

    return render_template('index.html', results=results, plot_urls=plot_urls, error_message=error_message, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)