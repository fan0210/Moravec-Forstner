#ifndef MORAVEC_FEATURE_DETECTOR_
#define MORAVEC_FEATURE_DETECTOR_

#include <opencv.hpp>
#include <cmath>

#include "FeatureDetector.h"

namespace fd
{
	class MoravecFeatureDetector : public FeatureDetector
	{
	public:
		MoravecFeatureDetector() = default;
		MoravecFeatureDetector(const int &cellSize, const int &blockSize, const int &thresh) :FeatureDetector(cellSize, blockSize), m_thresh(thresh) {}

		~MoravecFeatureDetector() = default;

		MoravecFeatureDetector &setThresh(const int &thresh) { this->m_thresh = thresh; return *this; }    //设置选取候选点的阈值
		const std::vector<KeyPoint> &detect(const cv::Mat &inputImg)override;

	private:
		int m_thresh;    //判断为候选点的阈值

		void detectCandidatePoints(const cv::Mat &)override;    //检测得到候选点
		void filterKeyPoints(const cv::Mat &)override;          //由候选点过滤得到特征点

		/*
		* 判断给定的点是否为窗口内兴趣值最大的点，(c, r)为窗口左上角坐标
		*/
		bool isPointInBlock(const KeyPoint &, const int &c, const int &r)const;
	};
}
#endif
