"""
Climate data generator module
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class ClimateDataGenerator:
    """Generate realistic synthetic climate and energy data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']['synthetic']
        self.start_year = self.data_config['start_year']
        self.end_year = self.data_config['end_year']
        self.countries = self.data_config['countries']
        
        # Set random seed if specified
        if 'random_seed' in self.data_config and self.data_config['random_seed'] is not None:
            np.random.seed(self.data_config['random_seed'])
    
    def generate_temperature_data(self) -> pd.DataFrame:
        """Generate global temperature anomaly data"""
        print("🌡️  Generating temperature data...")
        
        # Create monthly date range
        dates = pd.date_range(
            start=f'{self.start_year}-01-01',
            end=f'{self.end_year}-12-31',
            freq='MS'
        )
        
        # Base warming trend
        years = (dates.year - self.start_year).values
        base_trend = 0.8 * (years / 100)  # 0.8°C per century
        
        # Accelerated warming post-1970
        acceleration = np.where(
            dates.year >= 1970,
            0.5 * ((dates.year - 1970).values / 100),
            0
        )
        
        # Seasonality
        seasonality = 0.3 * np.sin(2 * np.pi * dates.month / 12 - np.pi/6)
        
        # Climate oscillations
        amo = 0.1 * np.sin(2 * np.pi * years / 60)  # Atlantic Multidecadal Oscillation
        pdo = 0.05 * np.sin(2 * np.pi * years / 20 + np.pi/4)  # Pacific Decadal Oscillation
        
        # Volcanic eruptions
        volcanic = self._generate_volcanic_eruptions(dates)
        
        # Random noise (increasing over time)
        noise_var = 0.05 + 0.1 * (years / max(years))
        noise = np.random.normal(0, noise_var, len(dates))
        
        # Combine all components
        anomaly = base_trend + acceleration + seasonality + amo + pdo + volcanic + noise
        
        return pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'anomaly': anomaly,
            'anomaly_smoothed': pd.Series(anomaly).rolling(12, center=True).mean(),
            'base_trend': base_trend,
            'acceleration': acceleration,
            'seasonality': seasonality,
            'amo': amo,
            'pdo': pdo,
            'volcanic': volcanic,
            'noise': noise
        })
    
    def _generate_volcanic_eruptions(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate volcanic cooling effects"""
        volcanic = np.zeros(len(dates))
        
        # Major historical eruptions with approximate cooling
        eruptions = {
            1883: -0.5,  # Krakatoa
            1902: -0.3,  # Santa María
            1912: -0.4,  # Novarupta
            1963: -0.3,  # Agung
            1982: -0.4,  # El Chichón
            1991: -0.5,  # Pinatubo
            2010: -0.2,  # Eyjafjallajökull
        }
        
        for year, cooling in eruptions.items():
            if self.start_year <= year <= self.end_year:
                # Find index for June of eruption year
                try:
                    idx = dates.get_loc(pd.Timestamp(f'{year}-06-01'))
                    # Apply exponential decay over 2 years
                    for i in range(idx, min(idx + 24, len(dates))):
                        months_since = i - idx
                        decay = np.exp(-months_since / 6)
                        volcanic[i] += cooling * decay
                except:
                    continue
        
        return volcanic
    
    def generate_emissions_data(self) -> pd.DataFrame:
        """Generate greenhouse gas emissions data"""
        print("🏭 Generating emissions data...")
        
        years = np.arange(self.start_year, self.end_year + 1)
        gases = ['CO2', 'CH4', 'N2O']
        sectors = ['Energy', 'Industry', 'Agriculture', 'Waste', 'Land Use Change']
        
        records = []
        
        # Country-specific emission profiles
        country_profiles = {
            'USA': {'growth': 1.5, 'peak_year': 2005, 'decline_rate': 0.02},
            'CHN': {'growth': 3.0, 'peak_year': 2030, 'decline_rate': 0.01},
            'IND': {'growth': 2.5, 'peak_year': 2040, 'decline_rate': 0.005},
            'EU27': {'growth': 1.2, 'peak_year': 1990, 'decline_rate': 0.015},
            'RUS': {'growth': 1.3, 'peak_year': 1990, 'decline_rate': 0.01},
            'BRA': {'growth': 1.8, 'peak_year': 2020, 'decline_rate': 0.008},
            'JPN': {'growth': 1.4, 'peak_year': 2013, 'decline_rate': 0.01},
            'AUS': {'growth': 1.6, 'peak_year': 2006, 'decline_rate': 0.008},
            'CAN': {'growth': 1.7, 'peak_year': 2007, 'decline_rate': 0.007},
            'MEX': {'growth': 2.0, 'peak_year': 2012, 'decline_rate': 0.006},
        }
        
        for year in years:
            for country in self.countries:
                profile = country_profiles.get(country, {'growth': 1.5, 'peak_year': 2020, 'decline_rate': 0.01})
                
                for gas in gases:
                    # Base emissions by gas type
                    base_emissions = {
                        'CO2': 1000,
                        'CH4': 50,
                        'N2O': 10
                    }[gas]
                    
                    # Calculate growth/decline
                    if year < profile['peak_year']:
                        # Growing phase
                        growth_factor = profile['growth'] ** (year - self.start_year)
                    else:
                        # Declining phase
                        years_since_peak = year - profile['peak_year']
                        decline_factor = (1 - profile['decline_rate']) ** years_since_peak
                        peak_emissions = base_emissions * (profile['growth'] ** (profile['peak_year'] - self.start_year))
                        growth_factor = peak_emissions * decline_factor / base_emissions
                    
                    for sector in sectors:
                        # Sector distribution
                        sector_share = {
                            'CO2': {'Energy': 0.65, 'Industry': 0.20, 'Agriculture': 0.05, 'Waste': 0.05, 'Land Use Change': 0.05},
                            'CH4': {'Energy': 0.30, 'Industry': 0.10, 'Agriculture': 0.40, 'Waste': 0.15, 'Land Use Change': 0.05},
                            'N2O': {'Energy': 0.10, 'Industry': 0.20, 'Agriculture': 0.60, 'Waste': 0.05, 'Land Use Change': 0.05},
                        }[gas][sector]
                        
                        # Add some randomness
                        randomness = np.random.uniform(0.9, 1.1)
                        
                        emissions = base_emissions * growth_factor * sector_share * randomness
                        
                        records.append({
                            'year': year,
                            'country': country,
                            'gas': gas,
                            'sector': sector,
                            'emissions_MT': max(emissions, 0),
                            'per_capita': emissions / np.random.uniform(10, 1000)  # Simplified per capita
                        })
        
        df = pd.DataFrame(records)
        
        # Calculate cumulative emissions
        for country in self.countries:
            for gas in gases:
                mask = (df['country'] == country) & (df['gas'] == gas)
                df.loc[mask, 'cumulative'] = df.loc[mask, 'emissions_MT'].cumsum()
        
        return df
    
    def generate_energy_data(self) -> pd.DataFrame:
        """Generate energy production data"""
        print("⚡ Generating energy data...")
        
        years = np.arange(max(1965, self.start_year), self.end_year + 1)
        sources = ['Coal', 'Oil', 'Gas', 'Nuclear', 'Hydro', 'Wind', 'Solar', 'Other Renewables']
        
        records = []
        
        # Country-specific energy profiles
        energy_profiles = {
            'USA': {'total': 4000, 'coal_share': 0.2, 'renewable_growth': 0.03},
            'CHN': {'total': 8000, 'coal_share': 0.6, 'renewable_growth': 0.05},
            'IND': {'total': 2000, 'coal_share': 0.5, 'renewable_growth': 0.04},
            'EU27': {'total': 3000, 'coal_share': 0.15, 'renewable_growth': 0.04},
            'RUS': {'total': 1500, 'coal_share': 0.3, 'renewable_growth': 0.02},
            'BRA': {'total': 600, 'coal_share': 0.1, 'renewable_growth': 0.03},
            'JPN': {'total': 1000, 'coal_share': 0.3, 'renewable_growth': 0.03},
            'AUS': {'total': 300, 'coal_share': 0.4, 'renewable_growth': 0.04},
            'CAN': {'total': 700, 'coal_share': 0.1, 'renewable_growth': 0.03},
            'MEX': {'total': 400, 'coal_share': 0.2, 'renewable_growth': 0.03},
        }
        
        for year in years:
            for country in self.countries:
                profile = energy_profiles.get(country, {'total': 1000, 'coal_share': 0.3, 'renewable_growth': 0.03})
                
                # Total energy growing over time
                total_energy = profile['total'] * (1.02 ** (year - 1965))
                
                # Time-dependent energy mix
                if year < 1980:
                    mix = {
                        'Coal': 0.4, 'Oil': 0.4, 'Gas': 0.1, 'Nuclear': 0.05,
                        'Hydro': 0.05, 'Wind': 0, 'Solar': 0, 'Other Renewables': 0
                    }
                elif year < 2000:
                    mix = {
                        'Coal': 0.35, 'Oil': 0.35, 'Gas': 0.15, 'Nuclear': 0.08,
                        'Hydro': 0.06, 'Wind': 0.005, 'Solar': 0.002, 'Other Renewables': 0.003
                    }
                elif year < 2020:
                    mix = {
                        'Coal': 0.3, 'Oil': 0.3, 'Gas': 0.2, 'Nuclear': 0.1,
                        'Hydro': 0.05, 'Wind': 0.02, 'Solar': 0.01, 'Other Renewables': 0.02
                    }
                else:
                    # Recent shift to renewables
                    renewable_share = min(0.3, 0.01 + profile['renewable_growth'] * (year - 2020))
                    mix = {
                        'Coal': profile['coal_share'] * (1 - renewable_share),
                        'Oil': 0.2 * (1 - renewable_share),
                        'Gas': 0.2 * (1 - renewable_share),
                        'Nuclear': 0.1,
                        'Hydro': 0.05,
                        'Wind': renewable_share * 0.4,
                        'Solar': renewable_share * 0.4,
                        'Other Renewables': renewable_share * 0.2
                    }
                
                # Normalize mix
                total = sum(mix.values())
                for source in mix:
                    mix[source] /= total
                
                # Generate values with some randomness
                for source in sources:
                    base_value = total_energy * mix[source]
                    randomness = np.random.uniform(0.95, 1.05)
                    value = base_value * randomness
                    
                    records.append({
                        'year': year,
                        'country': country,
                        'source': source,
                        'generation_TWh': max(value, 0),
                        'share': mix[source] * 100,
                        'carbon_intensity': self._get_carbon_intensity(source)
                    })
        
        df = pd.DataFrame(records)
        
        # Calculate renewable share
        renewable_sources = ['Hydro', 'Wind', 'Solar', 'Other Renewables']
        df['is_renewable'] = df['source'].isin(renewable_sources)
        
        return df
    
    def _get_carbon_intensity(self, source: str) -> float:
        """Get carbon intensity for energy source (gCO2/kWh)"""
        intensities = {
            'Coal': 820, 'Oil': 720, 'Gas': 490,
            'Nuclear': 12, 'Hydro': 24, 'Wind': 11,
            'Solar': 45, 'Other Renewables': 50
        }
        return intensities.get(source, 500)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all climate datasets"""
        return {
            'temperature': self.generate_temperature_data(),
            'emissions': self.generate_emissions_data(),
            'energy': self.generate_energy_data()
        }