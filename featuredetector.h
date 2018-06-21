#ifndef FEATURE_DETECTOR_H_
#define FEATURE_DETECTOR_H_

#include <opencv.hpp>

#include <vector>
#include <random>
#include <memory>

namespace fd 
{
	using uint_8 = unsigned char;

	/*
	* 特征检测基类
	*/
	class FeatureDetector
	{
	public:
		FeatureDetector() = default;
		FeatureDetector(const int &cellSize, const int &blockSize) :m_cellSize(cellSize), m_blockSize(blockSize) {}

		virtual ~FeatureDetector() = default;

		/*
		* 点特征类型，包含点坐标、点的id、以及点的信息比如兴趣值等
		*/
		class KeyPoint
		{
		public:
			cv::Point2f point;
			size_t id;
			std::vector<double> info;
		};

		virtual FeatureDetector &setWinSize(const int &cellSize, const int &blockSize)     //设置两个窗口大小，cellSize<=blockSize
		{
			m_cellSize = cellSize; 
			if (cellSize > blockSize)
				m_blockSize = cellSize;
			else
				m_blockSize = blockSize;

			return *this; 
		}

		/*
		* 特征检测接口，输入参数为需要进行检测的image，必须为三通道或者单通道
		*/
		virtual const std::vector<KeyPoint> &detect(const cv::Mat &inputImg) = 0;
		virtual const std::vector<KeyPoint> &getKeyPoints()const { return m_pts; }

        /*
		* 根据检测结果画出检测得到的特征点，输入图像须为三通道或单通道
		*/
		virtual void drawKeyPoints(const cv::Mat &inputImg, cv::Mat &outputImg)const;

	protected:
		int m_cellSize = 5;     //单元窗口的大小（例如5*5）
		int m_blockSize = 20;    //块窗口的大小（大于等于单元窗口）

		std::vector<KeyPoint> m_candidatePts;              //候选点
		std::vector<KeyPoint> m_pts;                       //检测得到的特征点，用getKeyPoints()获取

		virtual void detectCandidatePoints(const cv::Mat &) = 0;       //检测得到候选点
		virtual void filterKeyPoints(const cv::Mat &) = 0;             //过滤得到特征点
	};
}

namespace fd
{
	inline void FeatureDetector::drawKeyPoints(const cv::Mat &inputImg, cv::Mat &outputImg)const
	{
		outputImg = inputImg.clone();

		static std::default_random_engine e;
		static std::uniform_int_distribution<int> u(0, 255);

		if (outputImg.channels() == 1)
			cv::cvtColor(outputImg, outputImg, CV_GRAY2BGR);

		for (auto it = m_pts.cbegin(); it != m_pts.cend(); ++it)
			cv::circle(outputImg, it->point, 2, cv::Scalar(u(e), u(e), u(e)));
	}
}

#endif
