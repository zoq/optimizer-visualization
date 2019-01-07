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
#include <boost/asio/deadline_timer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <cstdlib>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>

using namespace cv;

using namespace ens;
using namespace ens::test;
using namespace ens::sfinae;
using namespace arma;
using boost::asio::ip::udp;
using boost::asio::deadline_timer;


class UDPReceiveClient
{
public:
  UDPReceiveClient(const udp::endpoint& listen_endpoint)
    : socket_(io_service_, listen_endpoint),
      deadline_(io_service_)
  {
    // No deadline is required until the first socket operation is started. We
    // set the deadline to positive infinity so that the actor takes no action
    // until a specific deadline is set.
    deadline_.expires_at(boost::posix_time::pos_infin);

    // Start the persistent actor that checks for deadline expiry.
    check_deadline();
  }

  std::size_t receive(const boost::asio::mutable_buffer& buffer,
      boost::posix_time::time_duration timeout, boost::system::error_code& ec)
  {
    // Set a deadline for the asynchronous operation.
    deadline_.expires_from_now(timeout);

    // Set up the variables that receive the result of the asynchronous
    // operation. The error code is set to would_block to signal that the
    // operation is incomplete. Asio guarantees that its asynchronous
    // operations will never fail with would_block, so any other value in
    // ec indicates completion.
    ec = boost::asio::error::would_block;
    std::size_t length = 0;

    // Start the asynchronous operation itself. The handle_receive function
    // used as a callback will update the ec and length variables.
    socket_.async_receive(boost::asio::buffer(buffer),
        boost::bind(&UDPReceiveClient::handle_receive, _1, _2, &ec, &length));

    // Block until the asynchronous operation has completed.
    do io_service_.run_one(); while (ec == boost::asio::error::would_block);

    return length;
  }

private:
  void check_deadline()
  {
    // Check whether the deadline has passed. We compare the deadline against
    // the current time since a new asynchronous operation may have moved the
    // deadline before this actor had a chance to run.
    if (deadline_.expires_at() <= deadline_timer::traits_type::now())
    {
      // The deadline has passed. The outstanding asynchronous operation needs
      // to be cancelled so that the blocked receive() function will return.
      //
      // Please note that cancel() has portability issues on some versions of
      // Microsoft Windows, and it may be necessary to use close() instead.
      // Consult the documentation for cancel() for further information.
      socket_.cancel();

      // There is no longer an active deadline. The expiry is set to positive
      // infinity so that the actor takes no action until a new deadline is set.
      deadline_.expires_at(boost::posix_time::pos_infin);
    }

    // Put the actor back to sleep.
    deadline_.async_wait(boost::bind(&UDPReceiveClient::check_deadline, this));
  }

  static void handle_receive(
      const boost::system::error_code& ec, std::size_t length,
      boost::system::error_code* out_ec, std::size_t* out_length)
  {
    *out_ec = ec;
    *out_length = length;
  }

private:
  boost::asio::io_service io_service_;
  udp::socket socket_;
  deadline_timer deadline_;
};

class UDPClient
{
public:
  UDPClient(const std::string& host, const std::string& port) : host(host), port(port)
  { }

  ~UDPClient()
  {
    // socket_.close();
  }

  size_t recv(const double x, const double y)
  {
    size_t recvSize = 0;
    try
    {
      boost::asio::io_service io_service;

      std::string localPort = "8023";
      if (port != "8037")
        localPort = "8024";

      udp::endpoint local_endpoint = boost::asio::ip::udp::endpoint(
      boost::asio::ip::address::from_string("127.0.0.1"), boost::lexical_cast<int>(localPort));

      UDPReceiveClient c(local_endpoint);

       boost::system::error_code ec;
       arma::Mat<int32_t> meta(1, 1);
       c.receive(boost::asio::buffer(meta.memptr(), sizeof(int32_t) * 1),
          boost::posix_time::seconds(10), ec);

      if (ec)
        return recvSize;

      size_t packetSize = 100 * sizeof(int32_t);
      arma::Mat<int32_t> data(meta(0), 1);

      size_t b = sizeof(int32_t) * data.n_elem;
      size_t idx = 0;
      while(b > 0)
      {
        if (b > packetSize)
        {
          const size_t count =  c.receive(boost::asio::buffer(data.memptr() + idx,
              packetSize), boost::posix_time::seconds(2), ec);
          idx += (count / sizeof(int32_t));
          b -= count;
          recvSize += count;
        }
        else
        {
          const size_t count =  c.receive(boost::asio::buffer(
              data.memptr() + idx, b), boost::posix_time::seconds(2), ec);
          idx += (count / sizeof(int32_t));
          b -= count;
          recvSize += count;
        }

        if (ec)
          return recvSize;
      }

      std::cout << "i" << std::endl;

      std::cout << x << " " << y << std::endl;

      std::cout << data << std::endl;
    }
    catch (std::exception& e)
    {
      return recvSize;
    }

    return recvSize;
  }

  size_t send(const arma::mat& data, const size_t functionID, const double x, const double y)
  {
    size_t recvSize = 0;

    boost::asio::io_service io_service;
    udp::socket socket_(io_service, udp::endpoint(udp::v4(), 0));

    udp::resolver resolver(io_service);
    udp::resolver::query query(udp::v4(), host, port);
    udp::resolver::iterator iter = resolver.resolve(query);
    udp::endpoint endpoint_ = *iter;

    meta[0] = functionID;
    meta[1] = data.n_elem;
    socket_.send_to(boost::asio::buffer(meta), endpoint_);
    usleep(4000);
    size_t packetSize = 100 * sizeof(double);

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
      usleep(4000);
    }

    recvSize = recv(x, y);
    socket_.close();

    return recvSize;
  }

private:
  boost::array<uint32_t, 2> meta;

  std::string host;
  std::string port;
};

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
    return function.Evaluate(coordinates, begin, batchSize);
  }

  double Evaluate(const arma::mat& coordinates)
  {
    return function.Evaluate(coordinates);
  }

  double EvaluateWithGradient(const arma::mat& coordinates,
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

    return result;
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
      data(0, i) = evaluateHistory[i];
      data(1, i) = normHistory[i];
      data.submat(2, i, 3, i) = coordinatesHistory[i];
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
    arma::Mat<int> state;
    size_t stateIndex;
    bool oc = state.load("state/state.csv", arma::csv_ascii);

    if (!oc)
    {
      state = arma::Mat<int>(2, 1);
      state(0) = 0;
      state(1) = 0;
      state.save("state/state.csv", arma::csv_ascii);
    }

    std::string port = "";
    if (state(0) == 0)
    {
      port = "8037";
      stateIndex = 0;
    }
    else if (state(1) == 0)
    {
      port = "8038";
      stateIndex = 1;
    }

    if (port != "")
    {
      size_t recvSize = 0;

      state(stateIndex) = 1;
      state.save("state/state.csv", arma::csv_ascii);

      UDPClient client("192.168.8.28", port);

      arma::uvec indices(3);
      indices(0) = 0;
      indices(1) = 2;
      indices(2) = 3;

      arma::mat coordinatesSub = coordinates.rows(indices);
      if (coordinatesSub.n_elem < 10000)
      {
        for (size_t r = 0; r < 4; ++r)
        {
          if (client.send(coordinatesSub, functionID, x, y) > 0)
            break;
        }
      }

      state.load("state/state.csv", arma::csv_ascii);
      state(stateIndex) = 0;
      state.save("state/state.csv", arma::csv_ascii);
    }
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
  IQN iqnOpt(stepSize, 1, iterations, 1e-9);
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
  Katyusha katyushaOpt(stepSize, parameterA, 1, iterations,
      parameterB, 1e-9, false);
  KatyushaProximal katyushaProximalOpt(stepSize, parameterA, 1, iterations,
      parameterB, 1e-9, false);
  SVRG svrgOpt(stepSize, 1, iterations, parameterB, 1e-9, false);
  SVRG_BB svrgBBOpt(stepSize, 1, iterations, parameterB, 1e-9, false,
      SVRGUpdate(), BarzilaiBorweinDecay(parameterC));
  SARAH sarahOpt(stepSize, 1, iterations, 0, 1e-9, false);
  SARAH_Plus sarahPlusOpt(stepSize, 1, iterations, 0, 1e-9, false);

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
    OptimizeFunction(iqnOpt, functionID, mode, x, y);
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
    OptimizeFunction(katyushaOpt, functionID, mode, x, y);
  }
  else if (optimizer == "katyushaproximal")
  {
    OptimizeFunction(katyushaProximalOpt, functionID, mode, x, y);
  }
  else if (optimizer == "svrg")
  {
    OptimizeFunction(svrgOpt, functionID, mode, x, y);
  }
  else if (optimizer == "svrgbb")
  {
    OptimizeFunction(svrgBBOpt, functionID, mode, x, y);
  }
  else if (optimizer == "sarah")
  {
    OptimizeFunction(sarahOpt, functionID, mode, x, y);
  }
  else if (optimizer == "sarahplus")
  {
    OptimizeFunction(sarahPlusOpt, functionID, mode, x, y);
  }

  return 0;
}
