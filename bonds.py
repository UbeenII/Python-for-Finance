from math import exp

class ZeroCouponBonds:
    def __init__(self,principal, maturity, interest_rate):
        self.principal= principal
        self.maturity = maturity
        self.interest_rate = interest_rate / 100

    def present_value( self,x ,n):
        return x / ( 1+ self.interest_rate)**n

    def calculate_price(self):
        return self.present_value(self.principal,self.maturity)

class CouponBonds:
    def __init__(self,principal, rate,maturity, interest_rate):
        self.principal = principal
        self.rate = rate /100
        self.maturity = maturity
        self.interest_rate = interest_rate / 100
    def present_value( self,x ,n):
        return x / ( 1+ self.interest_rate)**n
    def calculate_price(self):
        price = 0
        for t in ( 1, self.maturity+1):
            price = price+ self.present_value( self.principal*self.rate,t)

        price = price + self.present_value(self.principal,self.maturity)

        return price

class ContinuousDiscounting:
    def __init__(self,principal , maturity, interest_rate):
        self.principal=principal
        self.Maturity=maturity
        self.interest_rate=interest_rate
    def present_value(self):
        return self.principal * exp(-self.interest_rate * self.Maturity)



if __name__ == '__main__':

    bond= ZeroCouponBonds(1000,2,4)
    print( bond.calculate_price())

    bond = CouponBonds(1000,10,3,4)
    print( bond.calculate_price())