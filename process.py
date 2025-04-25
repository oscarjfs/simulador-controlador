import numpy as np
from scipy.integrate import solve_ivp
from pyXSteam.XSteam import XSteam

class ShellAndTubeHeatExchanger:
    """
    Base class for heat exchanger simulation.
    """

    def __init__(self, transfer_area:float=0.0, transfer_coefficient:float=0.0, cold_flow_rate:float=0.0, hot_flow_rate:float=0.0, hot_mass_flow_rate:float=0.0,
                 cold_temperature_in:float=0.0, hot_temperature_in:float=0.0, cold_specific_heat:float=0.0, hot_specific_heat:float=0.0,
                 cold_density:float=0.0, hot_density:float=0.0, cold_viscosity:float=0.0, hot_viscosity:float=0.0, cold_roughness:float=0.0,
                 hot_roughness:float=0.0, cold_length:float=0.0, hot_length:float=0.0, cold_diameter:float=0.0, hot_diameter:float=0.0,
                 number_of_tubes:int = 0, cold_pressure_in:float=0.0, hot_pressure_in:float=0.0,
                 cold_pressure_drop:float=0.0, hot_pressure_drop:float=0.0, flow_type:str='counterflow'):
        """
        Initializes the heat exchanger object.
        """
        self.A = transfer_area
        self.U = transfer_coefficient
        self.F_c = cold_flow_rate
        self.F_h = hot_flow_rate
        self.m_h = hot_mass_flow_rate
        self.m_c = cold_flow_rate * cold_density
        self.T_c_in = cold_temperature_in
        self.T_h_in = hot_temperature_in
        self.c_p_c = cold_specific_heat
        self.c_p_h = hot_specific_heat
        self.rho_c = cold_density
        self.rho_h = hot_density
        self.miu_c = cold_viscosity
        self.miu_h = hot_viscosity
        self.e_c = cold_roughness
        self.e_h = hot_roughness
        self.L_c = cold_length
        self.L_h = hot_length
        self.D_c = cold_diameter
        self.D_h = hot_diameter
        self.N = number_of_tubes
        self.P_c_in = cold_pressure_in
        self.P_h_in = hot_pressure_in
        self.delta_P_c = cold_pressure_drop
        self.delta_P_h = hot_pressure_drop
        self.flow_type = flow_type

        self.V_c = 0.0
        self.V_h = 0.0

        self.c_v_c = self.c_p_c
        self.l_h = 0.0
        self.l_c = 0.0

        self.T_c_out = self.T_c_in
        self.T_h_out = self.T_h_in
        self.lmtd = 0.0
        self.Q = 0.0

        self.td = 15 # oulet temperature dead time
    
    def calculate_logarithmic_mean_temperature_difference(self) -> float:
        """
        Calculates the logarithmic mean temperature difference.
        """

        if self.T_h_in <= self.T_c_in:
            raise ValueError("Invalid temperature values for logarithmic mean temperature difference calculation.")

        elif self.flow_type == 'counterflow':
            dT1 = self.T_h_out - self.T_c_in
            dT2 = self.T_h_in - self.T_c_out
        elif self.flow_type == 'parallelflow':
            dT1 = self.T_h_in - self.T_c_in
            dT2 = self.T_h_out - self.T_c_out
        else:
            raise ValueError("Invalid flow type. Use 'counterflow' or 'parallelflow'.")

        self.lmtd = (dT1 - dT2) / np.log(dT1 / dT2)

        return self.lmtd
    
    def calculate_heat_transfer(self) -> float:
        """
        Calculates the heat transfer rate.
        """
        self.Q = self.U * self.A * self.lmtd
        return self.Q
    
    def calculate_pressure_drop(self):
        """
        Calculates the pressure drop.
        """
        pass


class Water2SteamHeatExchanger(ShellAndTubeHeatExchanger):
    """
    Class for water to steam heat exchanger simulation.
    """
    def __init__(self, cold_flow_rate: float, hot_mass_flow_rate: float,
                 cold_temperature_in: float, hot_temperature_in: float, cold_pressure_in: float):
        super().__init__(cold_flow_rate=cold_flow_rate, hot_mass_flow_rate=hot_mass_flow_rate,
                         cold_temperature_in=cold_temperature_in, hot_temperature_in=hot_temperature_in,
                         cold_pressure_in=cold_pressure_in)
        
        self.steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS) # m/kg/sec/°C/bar/W
        self.V_c = 1 # m3
        self.update_hx_properties()

        self.F_c_s = self.F_c
        self.T_c_in_s = self.T_c_in
        self.m_h_s = self.m_h
        self.T_c_out_s = self.T_c_in

        self.update_hx_coefficients()

    def update_hx_coefficients(self) -> None:
        """
        Updates the heat exchanger coefficients.
        """
        self.C1 = self.c_p_c/self.c_v_c/self.V_c
        self.C2 = self.l_h/self.c_v_c/self.V_c
        self.C3 = self.C1 * (self.T_c_in_s - self.T_c_out)
        self.tao = 1/(self.C1*self.F_c_s)

    def update_hx_properties(self) -> None:
        """
        Updates the heat exchanger properties.
        """
        self.rho_c = round(self.steamTable.rho_pt(self.P_c_in, self.T_c_out),2)
        self.c_p_c = round(self.steamTable.Cp_pt(self.P_c_in, self.T_c_out),4)
        self.c_v_c = round(self.steamTable.Cv_pt(self.P_c_in, self.T_c_out),4)
        self.l_h = round(self.steamTable.hV_t(self.T_h_in)-self.steamTable.hL_t(self.T_h_in), 2)

    def update_stable_state(self) -> None:
        """
        Updates the stable state of the heat exchanger.
        """
        self.T_c_out_s = ((1/self.tao) * self.T_c_in_s + self.C3 * self.F_c_s + self.C2 * self.m_h_s) / (1/self.tao)

    def update_cold_inlet_temperature(self, T_c_in: float) -> None:
        """
        Updates the cold inlet temperature.
        """
        if self.T_c_in_s != T_c_in:
            self.T_c_in_s = self.T_c_in
            self.T_c_in = T_c_in

    def update_cold_flow(self, F_c: float) -> None:
        """
        Updates the cold flow.
        """
        if self.F_c_s != F_c:
            self.F_c_s = self.F_c
            self.F_c = F_c

    def update_steam_flow(self, m_h: float) -> None:
        """
        Updates the steam flow.
        """
        if self.m_h_s != m_h:
            self.m_h_s = self.m_h
            self.m_h = m_h

    def edo_t_c_out(self, t, T_c_out, T_c_in, F_c, m_v) -> float:
        """
        Calculates the derivative of the cold outlet temperature with respect to time.

        Args:
            t (float): Time.
            T_c_out (float): Cold outlet temperature.
            T_c_in (float): Cold inlet temperature.
            F_c (float): Cold flow rate.
            m_v (float): Steam mass flow rate.

        Returns:
            float: The derivative of the cold outlet temperature with respect to time.
        """

        dTcout_dt = 0

        if t > self.td:
            dTcout_dt = -(1/self.tao) * T_c_out + (1/self.tao) * T_c_in + self.C3 * F_c + self.C2 * m_v

        return dTcout_dt

    def calculate_future_temperature(self, t_span):
        """
        Calculates the future temperature of the cold outlet.
        """
        self.update_hx_properties()
        self.update_hx_coefficients()
        sol = solve_ivp(self.edo_t_c_out, t_span, [self.T_c_out], args=(self.T_c_in, self.F_c, self.m_h), method='RK45')
        self.T_c_out = sol.y[0][-1] * np.random.normal(1, np.sqrt(1e-6 * 1)) # Agregar ruido a la salida
        return sol.t[-1], sol.y[0][-1]

    def simulate(self, t_span) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates the heat exchanger.

        Args:
            t_span (tuple[float, float]): The time span for the simulation.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the time and the cold outlet temperature.
        """
        t0 = 0
        tf = t_span[-1] - t_span[0]
        t = np.linspace(t0, tf, 1000)
        y = np.zeros_like(t)
        y[0] = self.T_c_out
        for i in range(1, len(t)):
            t[i], y[i] = self.calculate_future_temperature((t[i-1], t[i]))

        return t  + t_span[0] , y
