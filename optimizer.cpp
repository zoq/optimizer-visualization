/**
 * @file optimizer.cpp
 * @author Marcus Edel
 */
#include <ensmallen.hpp>

#include <iostream>
#include <sstream>

#include "boost/lexical_cast.hpp"
#include <boost/array.hpp>
#include <boost/asio.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;

using namespace ens;
using namespace ens::test;
using namespace ens::sfinae;
using namespace arma;
using boost::asio::ip::udp;

class UDPClient
{
public:
  UDPClient(const std::string& host, const std::string& port) : host(host), port(port)
  { }

  ~UDPClient()
  {
    // socket_.close();
  }

  void recv(const double x, const double y)
  {
    boost::asio::io_service io_service;

    udp::endpoint local_endpoint = boost::asio::ip::udp::endpoint(
    boost::asio::ip::address::from_string("127.0.0.1"), boost::lexical_cast<int>("8023"));

    udp::socket socket(io_service);
    socket.open(udp::v4());
    socket.bind(local_endpoint);

   arma::Mat<int32_t> meta(1, 1);

    udp::endpoint sender_endpoint;
    size_t len = socket.receive_from(
        boost::asio::buffer(meta.memptr(), sizeof(int32_t) * 2), sender_endpoint);

    size_t packetSize = 20 * sizeof(int32_t);
    arma::Mat<int32_t> data(meta(0), 1);

    size_t b = sizeof(int32_t) * data.n_elem;
    size_t idx = 0;
    while(b > 0)
    {
      if (b > packetSize)
      {
        const size_t count = socket.receive_from(
          boost::asio::buffer(data.memptr() + idx, packetSize), sender_endpoint);
        idx += (count / sizeof(int32_t));
        b -= count;
      }
      else
      {
        const size_t count = socket.receive_from(
          boost::asio::buffer(data.memptr() + idx, b), sender_endpoint);
        idx += (count / sizeof(int32_t));
        b -= count;
      }
    }

    std::cout << "i" << std::endl;

    std::cout << x << " " << y << std::endl;

    std::cout << data << std::endl;
  }

  void send(const arma::mat& data, const size_t functionID, const double x, const double y)
  {
    boost::asio::io_service io_service;
    udp::socket socket_(io_service, udp::endpoint(udp::v4(), 0));

    udp::resolver resolver(io_service);
    udp::resolver::query query(udp::v4(), host, port);
    udp::resolver::iterator iter = resolver.resolve(query);
    udp::endpoint endpoint_ = *iter;

    meta[0] = functionID;
    meta[1] = data.n_elem;
    socket_.send_to(boost::asio::buffer(meta), endpoint_);

    size_t packetSize = 10 * sizeof(double);

    size_t u = 0;
    size_t b = sizeof(double) * data.n_elem;
    size_t idx = 0;
    while (b > 0)
    {
      u++;
      if (b > packetSize)
      {
        const size_t count = socket_.send_to(boost::asio::buffer(data.memptr() + idx, packetSize), endpoint_);
        idx += (count / sizeof(double));
        b -= count;
      }
      else
      {
        const size_t count = socket_.send_to(boost::asio::buffer(data.memptr() + idx, sizeof(double)), endpoint_);
        idx += (count / sizeof(double));
        b -= count;
      }
    }

    recv(x, y);
    socket_.close();
  }

private:
  boost::array<int, 2> meta;

  std::string host;
  std::string port;
};


// ENS_HAS_EXACT_METHOD_FORM(Evaluate, HasEvaluate)

ENS_HAS_MEM_FUNC(BatchSize, HasBatchSize);

template<class FunctionType>
class WrapperFunction
{
 public:
  WrapperFunction(FunctionType& function) : function(function) { }

  arma::mat& Coordinates() { return coordinates; }

  void Shuffle() { function.Shuffle(); }

  size_t NumFunctions() const { return function.NumFunctions(); }

  arma::mat GetInitialPoint() const { return function.GetInitialPoint(); }

  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    const double result = function.Evaluate(coordinates, begin, batchSize);

    // coordinatesHistory.push_back(coordinates);
    // evaluateHistory.push_back(result);

    return result;
  }

  double Evaluate(const arma::mat& coordinates)
  {
    return function.Evaluate(coordinates);

    // coordinatesHistory.push_back(coordinates);
    // evaluateHistory.push_back(result);
    // return result;
  }

  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize)
  {
    function.Gradient(coordinates, begin, gradient, batchSize);

    // Store results.
    const double result = function.Evaluate(coordinates);
    coordinatesHistory.push_back(coordinates);
    evaluateHistory.push_back(result);
    normHistory.push_back(arma::norm(gradient));
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    function.Gradient(coordinates, gradient);

    // Store results.
    const double result = function.Evaluate(coordinates);
    coordinatesHistory.push_back(coordinates);
    evaluateHistory.push_back(result);
    normHistory.push_back(arma::norm(gradient));
  }

  void Save(arma::mat& data)
  {
    data = arma::zeros(coordinatesHistory[0].n_elem + 2,
        coordinatesHistory.size());
    for (size_t i = 0; i < coordinatesHistory.size(); ++i)
    {
      // data(0, i) = i;
      // data.submat(1, i, 2, i) = coordinatesHistory[i];
      // data(data.n_rows - 1, i) = evaluateHistory[i];

      data(0, i) = evaluateHistory[i];
      data(1, i) = normHistory[i];
      data.submat(2, i, 3, i) = coordinatesHistory[i];

      // data.submat(1, i, 2, i) = coordinatesHistory[i];
      // data(data.n_rows - 1, i) = evaluateHistory[i];
    }
  }

 private:
  FunctionType& function;
  arma::mat coordinates;
  std::vector<arma::mat> coordinatesHistory;
  std::vector<double> evaluateHistory;
  std::vector<double> normHistory;
};

template<typename T, typename F>
inline typename std::enable_if<
    HasBatchSize<T, size_t&(T::*)()>::value, void>::type
SetBatchSize(T& optimizer, F& function)
{
  optimizer.BatchSize() = function.NumFunctions();
}

template<typename T, typename F>
inline typename std::enable_if<
    !HasBatchSize<T, size_t&(T::*)()>::value, void>::type
SetBatchSize(T& optimizer, F& function)
{
  /* Nothing to do here */
}

template<typename OptimizerType, typename FunctionType>
void OptimizeOptimizer(OptimizerType& optimizer,
                       FunctionType& f,
                       const size_t mode,
                       const double x,
                       const double y,
                       const size_t functionID)
{
  WrapperFunction<FunctionType> wf(f);

  SetBatchSize(optimizer, f);
  wf.Coordinates() = wf.GetInitialPoint();
  wf.Coordinates()(0) = x;
  wf.Coordinates()(1) = y;

  optimizer.Optimize(wf, wf.Coordinates());

  arma::mat coordinates;
  wf.Save(coordinates);

  std::cout.precision(3);
  std::cout.setf(ios::fixed);
  coordinates.t().raw_print(cout);

  if (mode == 1 || mode == 3)
  {
    UDPClient client("192.168.8.28", "8037");
    client.send(coordinates, functionID, x, y);
  }
}

template<typename OptimizerType>
void OptimizeFunction(OptimizerType& optimizer,
                      const size_t functionID,
                      const size_t mode,
                      const double x,
                      const double y)
{
  if (functionID == 0)
  {
    BoothFunction f;
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
  else if (functionID == 1)
  {
    StyblinskiTangFunction f(2);
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
  else if (functionID == 2)
  {
    BukinFunction f;
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
  else if (functionID == 3)
  {
    DropWaveFunction f;
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
  else if (functionID == 4)
  {
    McCormickFunction f;
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
  else if (functionID == 5)
  {
    RastriginFunction f(2);
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
  else if (functionID == 6)
  {
    SphereFunction f(2);
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
  else if (functionID == 7)
  {
    EasomFunction f;
    OptimizeOptimizer(optimizer, f, mode, x, y, functionID);
  }
}

long newmap(long x, long in_min, long in_max, long out_min, long out_max)
{
  if( x == in_max)
    return out_max;
  else if(out_min < out_max)
    return (x - in_min) * (out_max - out_min+1) / (in_max - in_min) + out_min;
  else
    return (x - in_min) * (out_max - out_min-1) / (in_max - in_min) + out_min;
}

int main(void)
{
  char* qs;
  qs = getenv("QUERY_STRING");

  // Some server don't provide QUERY_STRING if it's empty so avoid strdup()'ing
  // a NULL pointer here.
  char* cgiInput = strndup(qs ? qs : "", 600);

  std::string data(cgiInput);


  std::string baseData = data;

  std::replace(data.begin(), data.end(), '=', ' ');
  std::replace(data.begin(), data.end(), '&', ' ');
  std::stringstream ss(data);

  size_t mode;
  std::string parameter;
  std::string optimizer;
  size_t functionID;
  double x, y;
  size_t iterations;
  double stepSize;
  double parameterA, parameterB, parameterC, parameterD;

  std::cout << "Content-type: text/html" << std::endl;
  std::cout << "Cache-Control: max-age=3600" << std::endl;
  std::cout << std::endl << std::endl;

  ss >> parameter >> mode
     >> parameter >> optimizer
     >> parameter >> functionID
     >> parameter >> x
     >> parameter >> y
     >> parameter >> iterations
     >> parameter >> stepSize
     >> parameter >> parameterA
     >> parameter >> parameterB
     >> parameter >> parameterC
     >> parameter >> parameterD;

  if (iterations <= 0 || iterations > 20000)
    return 0;

  // Data not well-formatted.
  if (ss.fail())
    return 0;

  std::replace(baseData.begin(), baseData.end(), '=', '-');
  std::replace(baseData.begin(), baseData.end(), '&', '-');

  if (mode == 3)
  {
    cv::Mat surface, plane;

    if (functionID == 0)
    {
      surface = imread("img/0_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/0_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 1)
    {
      surface = imread("img/1_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/1_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 2)
    {
      surface = imread("img/2_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/2_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 3)
    {
      surface = imread("img/3_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/3_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 4)
    {
      surface = imread("img/4_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/4_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 5)
    {
      surface = imread("img/5_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/5_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 6)
    {
      surface = imread("img/6_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/6_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 7)
    {
      surface = imread("img/7_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/7_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }
    else if (functionID == 8)
    {
      surface = imread("img/8_surface.bmp", CV_LOAD_IMAGE_COLOR);
      plane = imread("img/8_plane.bmp", CV_LOAD_IMAGE_COLOR);
    }

    size_t fx = plane.rows / 500.0 * x;
    size_t fy = plane.cols / 500.0 * y;

    cv::Vec3b pA = surface.at<cv::Vec3b>(fy, fx);

    size_t i = 0, j = 0;
    bool check = false;

    int foffset = -1;

    int f = 0;
    while(f < 20)
    {
      for (i = 0; i < plane.rows; ++i)
      {
        for (j = 0; j < plane.cols; ++j)
        {
          cv::Vec3b pB = plane.at<cv::Vec3b>(j, i);
          if ((int)pA.val[0] == (int)pB.val[0] &&
              (int)pA.val[1] == (int)pB.val[1] &&
              (int)pA.val[2] == (int)pB.val[2])
          {
            check = true;
            break;
          }
        }

        if (check)
        {
          break;
        }
      }

      if (fx <= 0 || fy <= 0)
        foffset *= -1;

      if (i < (plane.rows - 5) && j < (plane.cols - 5) && i > 5 && j > 5)
        break;

      fx -= foffset;
      fy += (1 * foffset);
      pA = surface.at<cv::Vec3b>(fy, fx);

      f++;
    }

    if (functionID == 0)
    {
      x = (i - 164);
      y = (j - 143);
      x = newmap(x, 0, 733, -11, 11);
      y = newmap(y, 0, 737, -11, 11);
    }
    else if (functionID == 1)
    {
      x = (i - 32);
      y = (j - 7);
      x = newmap(x, 0, 1000, -5, 5);
      y = newmap(y, 0, 1005, -5, 5);
    }
    else if (functionID == 2)
    {
      x = (i - 32);
      y = (j - 7);
      x = newmap(x, 0, 1000, -10, 10);
      y = newmap(y, 0, 1005, -10, 10);
    }
    else if (functionID == 3)
    {
      x = (i - 32);
      y = (j - 7);
      x = newmap(x, 0, 1000, -6, 6);
      y = newmap(y, 0, 1005, -6, 6);
    }
    else if (functionID == 4)
    {
      x = (i - 32);
      y = (j - 7);
      x = newmap(x, 0, 1000, -4, 4);
      y = newmap(y, 0, 1005, -4, 4);
    }
    else if (functionID == 5)
    {
      x = (i - 32);
      y = (j - 7);
      x = newmap(x, 0, 1000, -5, 5);
      y = newmap(y, 0, 1005, -5, 5);
    }
    else if (functionID == 6)
    {
      x = (i - 32);
      y = (j - 7);
      x = newmap(x, 0, 1000, -6, 6);
      y = newmap(y, 0, 1005, -6, 6);
    }
    else if (functionID == 7)
    {
      x = (i - 32);
      y = (j - 7);
      x = newmap(x, 0, 1000, -20, 20);
      y = newmap(y, 0, 735, -20, 20);
    }

    x *= 1;
    y *= -1;
  }

  Adam adamOpt(stepSize, 1, parameterB, parameterC, parameterD, iterations,
      parameterA, false);
  RMSProp rmsPropOpt(stepSize, 1, parameterA, 1e-8, iterations, 1e-9, false);
  AdaDelta adaDeltaOpt(stepSize, 1, parameterA, 1e-8, iterations, 1e-9, false);
  AdaGrad adaGradOpt(stepSize, 1, 1e-8, iterations, 1e-9, false);
  CNE cneOpt(parameterA, iterations, stepSize, parameterB, parameterC, 0.1);
  SMORMS3 smormsOpt(stepSize, 1, 1e-16, iterations, 1e-9, false);
  // IQN iqnOpt(stepSize, 1, iterations, 1e-9);
  CMAES<> cmaesOpt(0, parameterA, parameterB, 1, iterations, 1e-9);
  AdaMax adaMaxOpt(stepSize, 1, parameterA, parameterB, 1e-8, iterations,
      1e-9, false);
  AMSGrad amsGradOpt(stepSize, 1, parameterA, parameterB, 1e-8, iterations,
      1e-9, false);
  Nadam nadamOpt(stepSize, 1, parameterA, parameterB, 1e-8, iterations,
      1e-9, false);
  StandardSGD sgdOpt(stepSize, 1, iterations, 1e-9, false);
  MomentumSGD sgdMomentumOpt(stepSize, 1, iterations, 1e-9, false,
      MomentumUpdate(parameterA));
  L_BFGS lbfgsOpt(stepSize, iterations, parameterA, 1e-4, parameterB, 1e-9);
  GradientDescent gdOpt(stepSize, iterations, 1e-9);
  ExponentialSchedule schedule;
  SA<ExponentialSchedule> saOpt(schedule, iterations, stepSize, parameterA,
      parameterB, 1e-9, 3, parameterC, parameterD, 0.3);
  SPALeRASGD<> spalerasgdOpt(stepSize, 1, iterations, 1e-4);
  // Katyusha katyushaOpt(stepSize, parameterA, 1, iterations,
  //     parameterB, 1e-9, false);
  // KatyushaProximal katyushaProximalOpt(stepSize, parameterA, 1, iterations,
  //     parameterB, 1e-9, false);
  SVRG svrgOpt(stepSize, 1, iterations, parameterB, 1e-9, false);
  SVRG_BB svrgBBOpt(stepSize, 1, iterations, parameterB, 1e-9, false,
      SVRGUpdate(), BarzilaiBorweinDecay(parameterC));
  // SARAH sarahOpt(stepSize, 1, iterations, 0, 1e-9, false);
  // SARAH_Plus sarahPlusOpt(stepSize, 1, iterations, 0, 1e-9, false);

  if (optimizer == "adam")
  {
    OptimizeFunction(adamOpt, functionID, mode, x, y);
  }
  else if (optimizer == "rmsprop")
  {
    OptimizeFunction(rmsPropOpt, functionID, mode, x, y);
  }
  else if (optimizer == "adadelta")
  {
    OptimizeFunction(adaDeltaOpt, functionID, mode, x, y);
  }
  else if (optimizer == "adagrad")
  {
    OptimizeFunction(adaGradOpt, functionID, mode, x, y);
  }
  else if (optimizer == "cne")
  {
    OptimizeFunction(cneOpt, functionID, mode, x, y);
  }
  else if (optimizer == "smorms")
  {
    OptimizeFunction(smormsOpt, functionID, mode, x, y);
  }
  else if (optimizer == "iqn")
  {
    // OptimizeFunction(iqnOpt, functionID, mode, x, y);
  }
  else if (optimizer == "cmaes")
  {
    OptimizeFunction(cmaesOpt, functionID, mode, x, y);
  }
  else if (optimizer == "adamax")
  {
    OptimizeFunction(adaMaxOpt, functionID, mode, x, y);
  }
  else if (optimizer == "amsgrad")
  {
    OptimizeFunction(amsGradOpt, functionID, mode, x, y);
  }
  else if (optimizer == "nadam")
  {
    OptimizeFunction(nadamOpt, functionID, mode, x, y);
  }
  else if (optimizer == "sgd")
  {
    OptimizeFunction(sgdOpt, functionID, mode, x, y);
  }
  else if (optimizer == "sgdmomentum")
  {
    OptimizeFunction(sgdMomentumOpt, functionID, mode, x, y);
  }
  else if (optimizer == "lbfgs")
  {
    OptimizeFunction(lbfgsOpt, functionID, mode, x, y);
  }
  else if (optimizer == "gradientdescent")
  {
    OptimizeFunction(gdOpt, functionID, mode, x, y);
  }
  else if (optimizer == "simulatedannealing")
  {
    OptimizeFunction(saOpt, functionID, mode, x, y);
  }
  else if (optimizer == "spalerasgd")
  {
    OptimizeFunction(spalerasgdOpt, functionID, mode, x, y);
  }
  else if (optimizer == "katyusha")
  {
    // OptimizeFunction(katyushaOpt, functionID, mode, x, y);
  }
  else if (optimizer == "katyushaproximal")
  {
    // OptimizeFunction(katyushaProximalOpt, functionID, mode, x, y);
  }
  else if (optimizer == "svrg")
  {
    OptimizeFunction(svrgOpt, functionID, mode, x, y);
  }
  else if (optimizer == "svrgbb")
  {
    OptimizeFunction(svrgBBOpt, functionID, mode, x, y);
  }
  // else if (optimizer == "sarah")
  // {
  //   OptimizeFunction(sarahOpt, functionID, mode, x, y);
  // }
  // else if (optimizer == "sarahplus")
  // {
  //   OptimizeFunction(sarahPlusOpt, functionID, mode, x, y);
  // }

  return 0;
}
