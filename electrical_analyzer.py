class ElectricalAnalyzer:
    def __init__(self, rated_current, unbalance_threshold=5.0, overload_tolerance=1.05):
        
        self.rated_current = float(rated_current)
        self.unbalance_threshold = float(unbalance_threshold)
        self.overload_tolerance = float(overload_tolerance)

    def analyze_currents(self, i_a, i_b, i_c):
        
        # Absolute values
        ia = abs(i_a)
        ib = abs(i_b)
        ic = abs(i_c)

        # Average Current
        i_avg = (ia + ib + ic) / 3.0

        # when motor off
        if i_avg == 0:
            return {
                "i_avg": 0.0,
                "unbalance_percent": 0.0,
                "unbalance_fault": False,
                "overload_fault": False
            }

        # apply NEMA 
        # Max Deviation
        dev_a = abs(ia - i_avg)
        dev_b = abs(ib - i_avg)
        dev_c = abs(ic - i_avg)
        max_deviation = max(dev_a, dev_b, dev_c)

        unbalance_percent = (max_deviation / i_avg) * 100.0
        is_unbalanced = unbalance_percent > self.unbalance_threshold

        #  Overload
        is_overloaded = i_avg > (self.rated_current * self.overload_tolerance)

        
        return {
            "i_avg": round(i_avg, 2),
            "unbalance_percent": round(unbalance_percent, 2),
            "unbalance_fault": is_unbalanced,
            "overload_fault": is_overloaded
        }

