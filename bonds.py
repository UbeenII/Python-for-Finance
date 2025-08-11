from math import exp

class ZeroCouponBonds:
    def __init__(self, principal, maturity_in, interest_rate):
        self.principal = principal
        self.maturity = maturity_in
        self.interest_rate = interest_rate / 100

    def present_value(self, x, n):
        return x / (1 + self.interest_rate) ** n

    def calculate_price(self):
        return self.present_value(self.principal, self.maturity)

class CouponBonds:
    def __init__(self, principal, rate, maturity_in, interest_rate):
        self.principal = principal
        self.rate = rate / 100
        self.maturity = maturity_in
        self.interest_rate = interest_rate / 100

    def present_value(self, x, n):
        return x / (1 + self.interest_rate) ** n

    def calculate_price(self):
        price = 0
        for t in range(1, self.maturity + 1):
            price += self.present_value(self.principal * self.rate, t)
        price += self.present_value(self.principal, self.maturity)
        return price

class ContinuousDiscounting:
    def __init__(self, principal, maturity_in, interest_rate):
        self.principal = principal
        self.maturity = maturity_in
        self.interest_rate = interest_rate

    def present_value(self):
        return self.principal * exp(-self.interest_rate * self.maturity)

def bond_price(yield_to_maturity, coupon, face_value_in, periods):
    price = 0
    for t in range(1, periods + 1):
        price += coupon / (1 + yield_to_maturity) ** t
    price += face_value_in / (1 + yield_to_maturity) ** periods
    return price

def differentiator(coupon, yield_to_maturity, face_value_in, periods):
    derivative = 0
    for t in range(1, periods + 1):
        derivative -= t * coupon / (1 + yield_to_maturity) ** (t + 1)
    derivative -= periods * face_value_in / (1 + yield_to_maturity) ** (periods + 1)
    return derivative

def newtonraphson(coupon, face_value_in, periods, target_price, initial_guess=0.05, tolerance=1e-6, max_iterations=100):
    ytm = initial_guess
    for i in range(max_iterations):
        price = bond_price(ytm, coupon, face_value_in, periods)
        diff = price - target_price
        if abs(diff) < tolerance:
            return ytm
        deriv = differentiator(coupon, ytm, face_value_in, periods)
        if deriv == 0:
            print("Derivative zero, stopping iteration")
            break
        ytm = ytm - diff / deriv
    return ytm


face_value = 1000
coupon_rate = 0.06
maturity = 3
market_price = 950
coupon_payment = face_value * coupon_rate



if __name__ == '__main__':
    bond = ZeroCouponBonds(1000, 2, 4)
    print("Zero Coupon Bond Price:", bond.calculate_price())

    bond = CouponBonds(1000, 10, 3, 4)
    print("Coupon Bond Price:", bond.calculate_price())
