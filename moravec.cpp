#include "moravec.h"

namespace fd
{
	void MoravecFeatureDetector::detectCandidatePoints(const cv::Mat &img)
	{
		for (auto r = 0; r < img.rows - m_cellSize + 1; ++r)
		{
			for (auto c = 0; c < img.cols - m_cellSize + 1; ++c)
			{
				double V[4];
				for (auto i = 0; i < m_cellSize - 1; ++i)
				{
					V[0] += pow(img.at<uint_8>(r + m_cellSize / 2, c + i) - img.at<uint_8>(r + m_cellSize / 2, c + i + 1), 2);
					V[1] += pow(img.at<uint_8>(r + i, c + i) - img.at<uint_8>(r + i + 1, c + i + 1), 2);
					V[2] += pow(img.at<uint_8>(r + m_cellSize - i - 1, c + i) - img.at<uint_8>(r + m_cellSize - i - 2, c + i + 1), 2);
					V[3] += pow(img.at<uint_8>(r + i, c + m_cellSize / 2) - img.at<uint_8>(r + i + 1, c + m_cellSize / 2), 2);
				}

				double min = 10000000000000000000;
				for (int i = 0; i < 4; ++i)
					min = min < V[i] ? min : V[i];

				if (min > m_thresh+0.00000000000000001)  
				{
					KeyPoint point = {
						cv::Point2f(c + m_cellSize / 2,r + m_cellSize / 2),
						(r + m_cellSize / 2)*img.cols + (c + m_cellSize / 2) + 1,
						{min}
					};

					m_candidatePts.push_back(point);
				}
			}
		}
	}

	void MoravecFeatureDetector::filterKeyPoints(const cv::Mat &img)
	{
		for (auto r = 0; r < img.rows - m_blockSize + 1; r += m_blockSize)
		{
			for (auto c = 0; c < img.cols - m_blockSize + 1; c += m_blockSize)
			{
				KeyPoint point = { cv::Point2f(0,0),0,{ -1 } };

				for (auto it = m_candidatePts.cbegin(); it != m_candidatePts.cend(); ++it)
					if (isPointInBlock(*it, c, r))
						if (it->info[0] > point.info[0])
							point = *it;

				if (point.info[0] != -1)
					m_pts.push_back(point);
			}
		}

		m_candidatePts.clear();
	}

	inline bool MoravecFeatureDetector::isPointInBlock(const KeyPoint &point, const int &c, const int &r)const
	{
		if (point.point.x >= c&&point.point.x < c + m_blockSize&&point.point.y >= r&&point.point.y < r + m_blockSize)
			return true;
		else
			return false;
	}

	const std::vector<MoravecFeatureDetector::KeyPoint> &MoravecFeatureDetector::detect(const cv::Mat &img)
	{
		if (img.channels() == 3)
		{
			cv::Mat grayImg;
			cv::cvtColor(img, grayImg, CV_BGR2GRAY);

			detectCandidatePoints(grayImg);
			std::cout << "提取到 " << m_candidatePts.size() << " 个候选点" << std::endl;
			filterKeyPoints(grayImg);
			std::cout << "过滤得到 " << m_pts.size() << " 个特征点" << std::endl;
		}
		else
		{
			detectCandidatePoints(img);
			std::cout << "提取到 " << m_candidatePts.size() << " 个候选点" << std::endl;
			filterKeyPoints(img);
			std::cout << "过滤得到 " << m_pts.size() << " 个特征点" << std::endl;
		}

		return m_pts;
	}
}