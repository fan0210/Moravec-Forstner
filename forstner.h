#ifndef FORSTNER_FEATURE_DECTOR_H_
#define FORSTNER_FEATURE_DECTOR_H_

#include <opencv.hpp>
#include <cmath>

#include "FeatureDetector.h"

namespace fd
{
	class ForstnerFeatureDetector : public FeatureDetector
	{
	public:
		ForstnerFeatureDetector() = default;
		ForstnerFeatureDetector(const int &cellSize, const int &blockSize, const double &thresh_q, const double &thresh_w)
			:FeatureDetector(cellSize, blockSize), m_thresh_q(thresh_q), m_thresh_w(thresh_w) {}

		~ForstnerFeatureDetector() = default;

		/*
		* 直接通过q和w设置阈值
		*/
		ForstnerFeatureDetector &setThresh(const double &Tq, const double &Tw)
		{
			m_thresh_q = Tq;
			m_thresh_w = Tw;

			return *this;
		}

		/*
		* 通过权平均值设置阈值
		*/
		ForstnerFeatureDetector &setThreshByWeightAverage(const double &Tq, const double &f)
		{
			m_thresh_q = Tq;
			m_f = f;

			m_isSetThreshByWeight = true;
			m_isSetThreshByWeightAverage = true;

			return *this;
		}

		/*
		* 通过权中值设置阈值
		*/
		ForstnerFeatureDetector &setThreshByWeightMedian(const double &Tq, const double &c)
		{
			m_thresh_q = Tq;
			m_c = c;

			m_isSetThreshByWeight = true;
			m_isSetThreshByWeightMedian = true;

			return *this;
		}

		const std::vector<KeyPoint> &detect(const cv::Mat &inputImg)override;
	private:
		double m_thresh_q = 0, m_thresh_w = 0;
		double m_w_average = 0, m_w_median = 0;
		double m_f = 0, m_c = 0;

		bool m_isSetThreshByWeight = false,
		m_isSetThreshByWeightAverage = false,
		m_isSetThreshByWeightMedian = false;

		struct Robert    //Robert梯度
		{
			Robert() = default;
			Robert(const double &gu_, const double &gv_) :gu(gu_), gv(gv_) {}

			double gu = 0;
			double gv = 0;
		};

		double tr(const cv::Mat_<double> &)const;       //求矩阵的迹
		void detectCandidatePoints(const cv::Mat &)override;          //检测特征点
		void filterKeyPoints(const cv::Mat &)override;                //过滤特征点

		/*
		* 判断给定的点是否为窗口内兴趣值最大的点，(c, r)为窗口左上角坐标
		*/
		bool isPointInBlock(const KeyPoint &, const int &c, const int &r)const;
	};
}

#endif
