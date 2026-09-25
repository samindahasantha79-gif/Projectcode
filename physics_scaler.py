class PhysicsScaler:
    def __init__(self, base_hp=2.0, base_rpm=1797.0):
        self.base_hp = base_hp
        self.base_rpm = base_rpm

    def scale_down_to_baseline(self, features, current_hp, current_rpm):
       
        # make ratio
        power_ratio = current_hp / self.base_hp
        speed_ratio = current_rpm / self.base_rpm
        
        # Amplitude Multiplier
        
        multiplier = power_ratio * (speed_ratio ** 2)
        
        if multiplier == 0:
            multiplier = 1.0 
            
        scaled_features = features.copy()
        
        # Scale down only amlitude features
        amplitude_features = ['max', 'min', 'mean', 'sd', 'rms']
        
        for feat in amplitude_features:
            if feat in scaled_features:
                scaled_features[feat] = scaled_features[feat] * multiplier
                
        
        return scaled_features

