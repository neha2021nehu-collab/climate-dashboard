import numpy as np
import pandas as pd
from datetime import datetime
import yaml
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ClimateDataGenerator:
    """Generate synthetic climate and energy datasets"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.countries = self.config['data']['countries']
        self.start_year = self.config['data']['start_year']
        self.end_year = self.config['data']['end_year']
        
    def generate_temperature_data(self) -> pd.DataFrame:
        """Generate monthly temperature anomaly data"""
        print("Generating temperature data...")
        
        # Create date range
        dates = pd.date_range(
            start=f'{self.start_year}-01-01',
            end=f'{self.end_year}-12-31',
            freq='M'
        )
        
        n_months = len(dates)
        
        # Create realistic warming trend
        years_from_start = (dates.year - self.start_year).values
        trend = 0.015 * years_from_start / 100  # 0.015°C per decade
        
        # Add acceleration after 1970
        acceleration_mask = dates.year >= 1970
        trend[acceleration_mask] += 0.025 * (dates.year[acceleration_mask] - 1970).values / 100
        
        # Add seasonality
        seasonality = 0.3 * np.sin(2 * np.pi * dates.month / 12 - np.pi/6)
        
        # Add multi-decadal oscillations (AMO/PDO like)
        amo = 0.1 * np.sin(2 * np.pi * years_from_start / 60)
        pdo = 0.05 * np.sin(2 * np.pi * years_from_start / 20 + np.pi/4)
        
        # Add noise (more in recent years due to better measurements)
        noise_scale = 0.1 + 0.05 * (years_from_start / max(years_from_start))
        noise = np.random.normal(0, noise_scale, n_months)
        
        # Combine components
        anomaly = trend + seasonality + amo + pdo + noise
        
        # Add volcanic eruptions (sharp cooling events)
        volcanic_years = [1883, 1902, 1912, 1963, 1982, 1991, 2010]
        for year in volcanic_years:
            if self.start_year <= year <= self.end_year:
                idx = (dates.year == year) & (dates.month.isin([6, 7, 8]))
                if idx.any():
                    anomaly[idx] -= np.random.uniform(0.3, 0.8, idx.sum())
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'anomaly': anomaly,
            'anomaly_smoothed': pd.Series(anomaly).rolling(12, center=True).mean(),
            'uncertainty': np.random.uniform(0.05, 0.15, n_months),
            'hemisphere': np.where(dates.month.isin([11, 12, 1, 2, 3, 4]), 'NH', 'SH')
        })
        
        return df
    
    def generate_emissions_data(self) -> pd.DataFrame:
        """Generate greenhouse gas emissions data"""
        print("Generating emissions data...")
        
        years = np.arange(self.start_year, self.end_year + 1)
        gases = ['CO2', 'CH4', 'N2O']
        sectors = ['Energy', 'Industry', 'Agriculture', 'Waste', 'Land Use']
        
        records = []
        
        for year in years:
            for country in self.countries:
                for gas in gases:
                    for sector in sectors:
                        # Create realistic trends
                        base = {
                            'CO2': 1000,
                            'CH4': 50,
                            'N2O': 10
                        }[gas]
                        
                        # Growth trends by country
                        growth_factors = {
                            'USA': 0.5 if year > 2005 else 1.0,
                            'CHN': 3.0 if year > 2000 else 1.0,
                            'IND': 2.5 if year > 1990 else 1.0,
                            'EU27': 0.7 if year > 1990 else 1.0,
                            'RUS': 0.6 if year > 1990 else 1.0,
                            'BRA': 1.2,
                            'JPN': 0.8
                        }
                        
                        # Historical increase then recent decrease for some
                        if year > 2015 and country in ['USA', 'EU27']:
                            trend = 0.95 ** (year - 2015)
                        elif year > 2020 and country == 'CHN':
                            trend = 1.02 ** (year - 2020)
                        else:
                            trend = 1.01 ** (year - self.start_year)
                        
                        emissions = base * growth_factors.get(country, 1.0) * trend
                        
                        # Add noise
                        emissions *= np.random.uniform(0.8, 1.2)
                        
                        records.append({
                            'year': year,
                            'country': country,
                            'gas': gas,
                            'sector': sector,
                            'emissions_MT': max(emissions, 0),
                            'cumulative': 0  # Will be calculated
                        })
        
        df = pd.DataFrame(records)
        
        # Calculate cumulative emissions
        for country in self.countries:
            for gas in gases:
                mask = (df['country'] == country) & (df['gas'] == gas)
                df.loc[mask, 'cumulative'] = df.loc[mask, 'emissions_MT'].cumsum()
        
        return df
    
    def generate_energy_data(self) -> pd.DataFrame:
        """Generate energy production mix data"""
        print("Generating energy data...")
        
        years = np.arange(1965, self.end_year + 1)
        sources = ['Coal', 'Oil', 'Gas', 'Nuclear', 'Hydro', 'Wind', 'Solar', 'Other Renewables']
        
        records = []
        
        for year in years:
            for country in self.countries:
                total_energy = np.random.uniform(500, 5000)
                
                # Time-dependent mix
                if year < 1980:
                    mix_weights = [0.4, 0.4, 0.1, 0.05, 0.05, 0, 0, 0]
                elif year < 2000:
                    mix_weights = [0.35, 0.35, 0.15, 0.08, 0.06, 0.005, 0.002, 0.003]
                elif year < 2010:
                    mix_weights = [0.3, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.02]
                elif year < 2020:
                    mix_weights = [0.25, 0.25, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05]
                else:
                    mix_weights = [0.2, 0.2, 0.2, 0.1, 0.05, 0.1, 0.1, 0.05]
                
                # Country-specific adjustments
                country_factors = {
                    'CHN': [0.6, 0.2, 0.05, 0.02, 0.1, 0.02, 0.01, 0.0],
                    'USA': [0.2, 0.3, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01],
                    'EU27': [0.15, 0.2, 0.2, 0.2, 0.1, 0.1, 0.05, 0.0],
                }
                
                if country in country_factors:
                    mix_weights = country_factors[country]
                
                # Normalize
                mix_weights = np.array(mix_weights)
                mix_weights = mix_weights / mix_weights.sum()
                
                # Generate values
                for source, weight in zip(sources, mix_weights):
                    value = total_energy * weight * np.random.uniform(0.9, 1.1)
                    
                    records.append({
                        'year': year,
                        'country': country,
                        'source': source,
                        'generation_TWh': max(value, 0),
                        'share': weight * 100
                    })
        
        return pd.DataFrame(records)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets"""
        return {
            'temperature': self.generate_temperature_data(),
            'emissions': self.generate_emissions_data(),
            'energy': self.generate_energy_data()
        }