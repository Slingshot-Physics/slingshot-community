#ifndef JANKLOGGERHEADER
#define JANKLOGGERHEADER

#include <chrono>
#include <iostream>
#include <fstream>

class JankLogger
{
   public:
      JankLogger(const std::string & log_name, const std::string & func_name);

      JankLogger(
         const std::string & log_dir,
         const std::string & log_name,
         const std::string & func_name
      );

      JankLogger(void);

      ~JankLogger(void);

      void StartTimer(int body_id_a, int body_id_b, unsigned int frame_count);

      void StopTimer(void);

      JankLogger & operator=(const JankLogger & j);

   private:

      int body_id_a_;

      int body_id_b_;

      unsigned int frame_count_;

      std::string log_dir_;

      std::string log_name_;

      std::string func_name_;

      std::chrono::high_resolution_clock::time_point start_time_;

      std::ofstream log_file_;

      void AddFolder(std::string & log_dir);
};

#endif
