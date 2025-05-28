#include <iostream>
#include <vector>
#include <Eigen/Dense>

template<class T>
class Z_spline
{
protected:
    int m; // spline ord
    std::vector<T> X; // x coordinates of points
    std::vector<T> y; // y coordinates of points
    std::vector<Eigen::MatrixXd> derivatives_at_points; // derivatives up to m-1 ord in each point

    double factorial(int x)
        const {
        int res = 1;
        while (x > 0)
        {
            res *= x;
            --x;
        }
        return res;
    }

    // inverse of vandermonde matrix for point at index idx_of_point
    Eigen::MatrixXd V_inv(std::vector<T>& X_window, int idx_of_point)
        const {
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(X_window.size(), X_window.size());

        for (int l = 0; l < X_window.size(); ++l)
        {
            for (int p = 0; p < X_window.size(); ++p)
            {
                V(l, p) = std::pow((X_window[l] - X[idx_of_point]), p);
            }
        }

        return V.inverse();
    }

    //calcute amount of points at left and right of the point for derivative calculation
    std::pair<int, int> points_left_right(int idx_of_point)
        const {
        int amount_of_points_left = std::min(m - 1, idx_of_point);
        int amount_of_points_right = std::min(m - 1, int(X.size()) - idx_of_point - 1);

        if (amount_of_points_right < amount_of_points_left)
            amount_of_points_left += amount_of_points_left - amount_of_points_right;
        else
            amount_of_points_right = (m - 1 - amount_of_points_left) + m - 1;

        return std::make_pair(amount_of_points_left, amount_of_points_right);
    }

    Eigen::MatrixXd calculate_der_matrix(int idx_of_point)
        const {
        int amount_of_points_left = std::min(m - 1, idx_of_point);
        int amount_of_points_right = std::min(m - 1, int(X.size()) - idx_of_point - 1);

        if (amount_of_points_right < amount_of_points_left)
            amount_of_points_left += amount_of_points_left - amount_of_points_right;
        else
            amount_of_points_right = (m - 1 - amount_of_points_left) + m - 1;

        //if (idx_of_zero < idx_of_point - amount_of_points_left || idx_of_point + amount_of_points_right + 1 < idx_of_zero)
            //return Eigen::MatrixXd::Zero(2 * m - 1, 1);

        std::vector<T> der_window = std::vector<T>(X.begin() + idx_of_point - amount_of_points_left, X.begin() + idx_of_point + amount_of_points_right + 1);

        //Eigen::MatrixXd y_window = Eigen::MatrixXd::Zero(2 * m - 1, 1);
        //y_window(idx_of_zero - idx_of_point + amount_of_points_left, 0) = 1;

        return V_inv(der_window, idx_of_point);
    }

    // functions for hermite interpolation calculation
    double lk(double x, int k, int j)
        const {
        return k == 0 ? ((x - X[j + 1]) / (X[j] - X[j + 1])) : ((x - X[j]) / (X[j + 1] - X[j]));
    }

    double l_0m_taylor(double x, int j, int p)
        const {
        double res = 1;
        double nom = 1;
        double denom = 1;

        for (int i = 1; i <= m - p - 1; i++)
        {
            nom *= (m + i - 1) * (x - X[j]) * (-1);
            denom *= i * (X[j] - X[j + 1]);
            res += nom / denom;
        }

        return res;
    }

    double l_1m_taylor(double x, int j, int p)
        const {
        double res = 1;
        double nom = 1;
        double denom = 1;

        for (int i = 1; i <= m - p - 1; i++)
        {
            nom *= (m + i - 1) * (x - X[j + 1]);
            denom *= i * (X[j] - X[j + 1]);
            res += nom / denom;
        }

        return res;
    }

    double B_p0(int p, int j, double x)
        const {
        return pow(x - X[j], p) * l_0m_taylor(x, j, p) * std::pow(lk(x, 0, j), m) / factorial(p);
    }

    double B_p1(int p, int j, double x)
        const {
        return pow(x - X[j + 1], p) * l_1m_taylor(x, j, p) * pow(lk(x, 1, j), m) / factorial(p);
    }

    Eigen::MatrixXd get_derivative_at_point(int j, int idx_of_zero)
        const {
        std::pair<int, int> window = points_left_right(j); // index of start and end of points slice for point

        if (idx_of_zero < j - window.first || j + window.second < idx_of_zero)
            return Eigen::MatrixXd::Zero(2 * m - 1, 1);

        Eigen::MatrixXd y_der_j = Eigen::MatrixXd::Zero(2 * m - 1, 1);
        y_der_j(idx_of_zero - j + window.first, 0) = 1;

        return derivatives_at_points[j] * y_der_j;
    }

    //calculate value of Z-spline basis function at specified interval and basis point
    double Z_(double x, int interval, int idx_of_zero)
        const {
        double res = 0;

        Eigen::MatrixXd der_j = get_derivative_at_point(interval, idx_of_zero);
        Eigen::MatrixXd der_j_next = get_derivative_at_point(interval + 1, idx_of_zero);

        for (int p = 0; p <= m - 1; p++)
        {
            res += factorial(p) * der_j(p, 0) * B_p0(p, interval, x) + factorial(p) * der_j_next(p, 0) * B_p1(p, interval, x);
        }

        return res;
    }

    // find start of interval where points is located
    int find_interval(double x)
        const {
        int left = 0;
        int right = X.size() - 1;

        while (left < right)
        {
            int mid = left + (right - left) * 0.5;
            if (X[mid] <= x) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }

        return left - 1;
    }

    // calculate value of Z-spline at point
    double val_at_x(double x)
        const {
        int j = find_interval(x);
        double res = 0;

        for (int i = 0; i <= X.size() - 1; i++)
        {
            res += y[i] * Z_(x, j, i);
        }
        return res;
    }
public:
    Z_spline(int m, std::vector<T>& X, std::vector<T>& y) : m(m), X(X), y(y)
    {
        for (int i = 0; i < X.size(); i++)
        {
            derivatives_at_points.push_back(calculate_der_matrix(i));
        }
    }

    double operator()(double x)
    const {
        return val_at_x(x);
    }

    std::vector<double> operator()(std::vector<double> x)
        const {
        std::vector<double> res;

        for (int i = 0; i < x.size(); i++)
            res.push_back(val_at_x(x[i]));

        return res;
    }
};
