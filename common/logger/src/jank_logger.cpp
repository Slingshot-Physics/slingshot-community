#include "jank_logger.hpp"

#include "folders.hpp"

JankLogger::JankLogger(
   const std::string & log_name, const std::string & func_name
)
: body_id_a_(-1)
, body_id_b_(-1)
, frame_count_(0)
, log_dir_(".")
, log_name_(log_name)
, func_name_(func_name)
{
   log_file_.open(log_name_.c_str(), std::ofstream::out);
}

JankLogger::JankLogger(void)
: body_id_a_(-1)
, body_id_b_(-1)
, frame_count_(0)
, log_dir_("")
, log_name_("")
, func_name_("")
{

}

JankLogger::JankLogger(
   const std::string & log_dir,
   const std::string & log_name,
   const std::string & func_name
)
: body_id_a_(-1)
, body_id_b_(-1)
, frame_count_(0)
, log_dir_(log_dir)
, log_name_(log_name)
, func_name_(func_name)
{
   std::cout << "Adding logger " << log_name_ << "\n";
   AddFolder(log_dir_);
   log_name_ = log_dir_ + "/" + log_name_;
   log_file_.open(log_name_.c_str(), std::ofstream::out);
}

JankLogger::~JankLogger(void)
{
   log_file_.close();
}

void JankLogger::StartTimer(
   int body_id_a, int body_id_b, unsigned int frame_count
)
{
   start_time_ = std::chrono::high_resolution_clock::now();
   frame_count_ = frame_count;
   body_id_a_ = body_id_a;
   body_id_b_ = body_id_b;
}

void JankLogger::StopTimer(void)
{
   std::chrono::duration<double, std::milli> diff = std::chrono::high_resolution_clock::now() - start_time_;
   log_file_ << "frame: " << frame_count_ << " body_id_a: " << body_id_a_ << " body_id_b: " << body_id_b_ << " func: " << func_name_ << " dt_ms: " << diff.count() << "\n";
}

JankLogger & JankLogger::operator=(const JankLogger & j)
{
   body_id_a_ = j.body_id_a_;
   body_id_b_ = j.body_id_b_;
   frame_count_ = j.frame_count_;
   log_name_ = j.log_name_;
   func_name_ = j.func_name_;

   log_file_.open(log_name_.c_str(), std::ofstream::out);

   return *this;
}

void JankLogger::AddFolder(std::string & log_dir)
{
   if (folders::dir_exists(log_dir))
   {
      std::cout << "Folder " << log_dir << " already exists, so I won't try to remake it\n";
   }
   else
   {
      // int err = mkdir(log_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      int err = folders::mkdir(log_dir);
      if (err == -1)
      {
         std::cout << "Couldn't make directory " << log_dir << " dumping log files to local dir\n";
         log_dir = ".";
      }
      else
      {
         std::cout << "Made the directory " << log_dir << "\n";
      }
   }
}
