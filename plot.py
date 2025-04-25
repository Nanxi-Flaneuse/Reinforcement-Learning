import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as st

# helper function
def get_slope(x, y, type = 'linear'):
        # Number of data points
        n = len(x)

        # Calculate sums needed for slope and intercept
        
        if type == 'exponential':
            sum_y = np.sum(np.log(y))
            sum_xy = np.sum(x * np.log(y))
            sum_xx = np.sum(x**2)
            sum_x = np.sum(x)
        elif type == 'linear':
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_xx = np.sum(x**2)
            sum_x = np.sum(x)
        elif type == 'log':
            sum_y = np.sum(np.log10(y))
            sum_xy = np.sum(np.log10(x) * np.log10(y))
            sum_xx = np.sum(np.log10(x)**2)
            sum_x = np.sum(np.log10(x))
        

        

        # Calculate the slope (b)
        b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)

        # Calculate the y-intercept (a)
        a = (sum_y - b * sum_x) / n

        # print(f"Slope (b): {b}")
        # print(f"Y-intercept (a): {a}")

        if type == 'linear':
            print('b',str(b),'a',str(a))
            return b, a
        elif type == 'exponential':
            print('b',str(b),'a',str(np.exp(a)))
            return b, np.exp(a)
        elif type == 'log':
            print('b',str(b),'log_a',str(a))
            return b, a
# 11.2.2
def f_11_2_2():
    x_values = np.array([0, 0.5, 1, 2, 3, 4, 5,6,7,8])
    y_values = np.array([104.6,104.1,104.4,105,106,106.8,107.7,108.7,110.6,112.1])

    b,a = get_slope(x_values,y_values)

    # Create the regression line
    regression_line = b * x_values + a

    # Plot the original data points
    # plt.scatter(x_values, y_values, label='Data Points')

    # Plot the regression line
    plt.plot(x_values, regression_line, color='red', label=f'Regression Line: y = {b:.2f}x + {a:.2f}')

    # Add labels and title
    plt.xlabel('age')
    plt.ylabel('proof')
    plt.title('Least Squares Regression Line')

    # Add legend
    plt.legend()

    # plt.plot(x_values, y_values)
    # plt.xlabel("Age")
    # plt.ylabel("Proof")
    # plt.title("Simple Line Plot")
    plt.savefig('stats_graphs/11_2_2.png')

# 11.2.8
def f_11_2_8():
    # Given data
    plant_cover_diversity = [0.90, 0.76, 1.67, 1.44, 0.20, 0.16, 1.12, 1.04, 0.48, 1.33, 1.10, 1.56, 1.15]
    bird_species_diversity = [1.80, 1.36, 2.92, 2.61, 0.42, 0.49, 1.90, 2.38, 1.24, 2.80, 2.41, 2.80, 2.16]

    # Convert lists to numpy arrays for easier calculations
    x = np.array(plant_cover_diversity)
    y = np.array(bird_species_diversity)

    b,a = get_slope(x,y)
    print('slope',str(b),'intercept',str(a))
    predicts = b*x + a
    residuals = predicts - y
    plt.figure(figsize=(8, 6))
    plt.scatter(x, residuals, color='blue', label='Residuals', marker='o')
    plt.axhline(0, color='black',linewidth=1)  # Adds a horizontal line at y=0 for reference
    plt.title('Dot Plot of Residuals')
    plt.xlabel('Plant Cover Diversity (x)')
    plt.ylabel('Residuals (y - y_pred)')
    plt.grid(True)
    # plt.show()
    plt.savefig('stats_graphs/11_2_8.png')

def f_11_2_22():
     # Age in years (x)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Suggested retail price (y)
    y = np.array([14680, 12150, 11215, 10180, 9230, 8455, 7730, 6825, 6135, 5620])
    b,a = get_slope(x,y,'exponential')
    regression_line = a*np.exp(b*x)
        # Plot the regression line
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, regression_line, color='red', label=f'Regression Line')

    # Add labels and title
    plt.xlabel('age')
    plt.ylabel('suggested retail price')
    plt.title('Least Squares Regression Line')

    # Add legend
    plt.legend()

    # plt.plot(x_values, y_values)
    # plt.xlabel("Age")
    # plt.ylabel("Proof")
    # plt.title("Simple Line Plot")
    plt.savefig('stats_graphs/11_2_22.png')
    print('11 year old toyota worth:',str(a*np.exp(b*11)))

def f_11_2_26():

    # Locomotion Begins (x)
    x = np.array([360, 165, 21, 23, 11, 18, 18, 150, 45, 45, 18])

    # Play Begins (y)
    y = np.array([90, 105, 21, 26, 14, 28, 21, 105, 68, 75, 46])
    b,log_a = get_slope(x,y,'log')
    # Define the power-law function y = a * x^b
    # def power_law(x, a, b):
    #     return a * x**b

    # # Perform the curve fitting
    # params, covariance = curve_fit(power_law, x, y)

    # # Calculate the fitted y values (the power-law curve)
    # y_pred = power_law(x, a, b)

    # # Plotting the data and the power-law regression curve
    # plt.scatter(x, y, color='blue', label='Data Points')  # Scatter plot of data
    # plt.plot(x, y_pred, color='red', label=f'Power-Law Fit: y = {a:.2f} * x^{b:.2f}')  # Power-law regression curve
    # plt.xlabel('Locomotion (days)')
    # plt.ylabel('Playfulness (days)')
    # plt.title('Power-Law Curve Fitting')
    # plt.legend()
    # plt.savefig('stats_graphs/11_2_26.png')
    regression_line = b*np.log10(x) + log_a
        # Plot the regression line
    plt.scatter(np.log(x), np.log(y), label='Data Points')
    plt.plot(np.log(x), regression_line, color='red', label=f'Regression Line')

    # Add labels and title
    plt.xlabel('locomotion')
    plt.ylabel('playfulness')
    plt.title('Least Squares Regression Line')

    # Add legend
    plt.legend()
    plt.savefig('stats_graphs/11_2_26.png')   


def f_11_3_2():

    # Spending per pupil (in 1000s), x
    x = np.array([10.0, 10.2, 10.2, 10.3, 10.3, 10.8, 11.0, 11.0, 11.2, 11.6, 12.1, 12.3, 12.6, 12.7, 12.9, 13.0, 13.9, 14.5, 14.7, 15.5, 16.4, 17.5, 18.1, 20.8, 22.4, 24.0])
    mean = np.mean(x)
    diff = (x - mean)**2
    thingy = np.sqrt(np.sum(diff))
    print(0.412 - 2.064*11.78848/thingy,0.412 + 2.064*11.78848/thingy)
    # Graduation rate, y
    y = np.array([88.7, 93.2, 95.1, 94.0, 88.3, 89.9, 67.7, 90.2, 93.5, 75.2, 84.6, 85.0, 94.8, 56.1, 54.4, 97.9, 83.0, 94.0, 91.4, 94.2, 97.2, 94.4, 78.6, 87.6, 93.3, 92.3])
    # Create the regression line
    regression_line = 81.088 + 0.412 * x

    # Plot the original data points
    # plt.scatter(x_values, y_values, label='Data Points')
    plt.scatter(x, y, label='Data Points')
    # Plot the regression line
    plt.plot(x, regression_line, color='red', label=f'Regression Line: y = {0.412:.2f}x + {81.088:.2f}')

    # Add labels and title
    plt.xlabel('spending')
    plt.ylabel('graduation')
    plt.title('Least Squares Regression Line')

    # Add legend
    plt.legend()

    # plt.plot(x_values, y_values)
    # plt.xlabel("Age")
    # plt.ylabel("Proof")
    # plt.title("Simple Line Plot")
    plt.savefig('stats_graphs/11_3_2.png')

def f_11_4_14():
    # Given sums
    sum_x = 994.7700
    sum_x_squared = 28462.1047
    sum_y = 254.6900
    sum_y_squared = 1816.1417
    sum_xy = 7051.2633
    n = 36
    print((n*sum_xy - sum_x*sum_y)/np.sqrt((n*sum_x_squared - sum_x**2)*(n*sum_y_squared - sum_y**2)))

def f_11_5_4():
    E = 11 - 0.6*np.sqrt(2.6/1.2)
    pho = np.sqrt((1-0.36)*2.6**2)
    # print(E, pho)

    def normal_probability(lower_bound, upper_bound, mean, std_dev):
        """
        Calculates the probability within a range for a normal distribution.

        Args:
            lower_bound: The lower limit of the range.
            upper_bound: The upper limit of the range.
            mean: The mean of the normal distribution.
            std_dev: The standard deviation of the normal distribution.

        Returns:
            The probability within the specified range.
        """
        return st.norm.cdf(upper_bound, loc=mean, scale=std_dev) - st.norm.cdf(lower_bound, loc=mean, scale=std_dev)
    
    # print(normal_probability(10,10.5, E, pho))
    print(normal_probability(10,10.5, E, pho/2))

def f_11_5_8():
    # Sums for x and y
    sum_x = 2458
    sum_y = 4097

    # Sums for x^2 and y^2
    sum_x_squared = 444118
    sum_y_squared = 1262559

    # Sum for x*y
    sum_xy = 710499
    n = 14
    r = (n*sum_xy - sum_x*sum_y)/np.sqrt((n*sum_x_squared - sum_x**2)*(n*sum_y_squared - sum_y**2))
    t = np.sqrt(12)*r/np.sqrt(1-r**2)
    print(r,t)


if __name__ == '__main__':
    # f_11_3_2()
    f_11_5_8()

    

