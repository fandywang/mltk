// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The log facility, which makes it easy to leave of trace of your
// program.  The logs are classified according to their severity
// levels.  Logs of the level FATAL will cause a segmentation fault,
// which makes the debugger to keep track of the stack.
//
// Examples:
//   LOG(INFO) << iteration << "-th iteration ..." << std::endl;
//   LOG(FATAL) << "Probability value < 0 " << prob_value << std::endl;
//

#ifndef MLTK_COMMON_LOGGING_H_
#define MLTK_COMMON_LOGGING_H_

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>

// The CHECK_xxxx facilities, which generates a segmentation fault
// when a check is failed.  If the program is run within a debugger,
// the segmentation fault makes the debugger keeps track of the stack,
// which provides the context of the fail.
//

#define CHECK(a) if (!(a)) {                                            \
    std::cerr << "CHECK failed "                                        \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n";                            \
    exit(1);                                                            \
  }                                                                     \

#define CHECK_EQ(a, b) if (!((a) == (b))) {                             \
    std::cerr << "CHECK_EQ failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    exit(1);                                                            \
  }                                                                     \

#define CHECK_GT(a, b) if (!((a) > (b))) {                              \
    std::cerr << "CHECK_GT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    exit(1);                                                            \
  }                                                                     \

#define CHECK_LT(a, b) if (!((a) < (b))) {                              \
    std::cerr << "CHECK_LT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    exit(1);                                                            \
  }                                                                     \

#define CHECK_GE(a, b) if (!((a) >= (b))) {                             \
    std::cerr << "CHECK_GE failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    exit(1);                                                            \
  }                                                                     \

#define CHECK_LE(a, b) if (!((a) <= (b))) {                             \
    std::cerr << "CHECK_LE failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    exit(1);                                                            \
  }                                                                     \
                                                                        \


enum LogSeverity { INFO, WARNING, ERROR, FATAL };

class Logger {
 public:
  Logger(LogSeverity ls, const std::string& file, int line)
      : ls_(ls), file_(file), line_(line) {
    switch (ls) {
    case INFO:
      ls_str_ = "INFO";
      break;
    case WARNING:
      ls_str_ = "WARNING";
      break;
    case ERROR:
      ls_str_ = "ERROR";
      break;
    case FATAL:
      ls_str_ = "FATAL";
      break;
    default:
      ls_str_ = "UNKNOWN";
    }
  }
  ~Logger() {
    if (ls_ == FATAL) {
      exit(1);
    }
  }

  std::ostream& stream() const {
    time_t rawtime;
    time(&rawtime);
    struct tm* timeinfo;
    timeinfo = localtime(&rawtime);
    char buffer[80];
    strftime(buffer, 80, "%Y/%m/%d %H:%S:%M", timeinfo);

    return std::cerr << buffer << " " << file_ << " (Line: " << line_ << ") ["
        << ls_str_ << "]:";
  }

 private:
  LogSeverity ls_;
  std::string ls_str_;
  std::string file_;
  int line_;
};

#define LOG(ls) Logger(ls, __FILE__, __LINE__).stream()

#endif  // MLTK_COMMON_LOGGING_H_

