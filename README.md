# Policy_Gradient_ETF_Portfolio_Manager

Deterministic Policy Gradient Algorithm을 이용하여
ETF 상품 15개의 포트폴리오를 구성하는 프로그램

거래 대상 종목은 asset_name.csv 에서 확인할 수 있다. 


# TODO (단기적)
1. ~~loss(critic) 변경 (sharpe ratio or mse between portfolio value gain and optimal) 현재는 단순히 portfolio value gain이용중~~ DONE
1. 초기값에 따라 policy 수렴이 달라지는 문제 해결 ~~(앙상블 or 모델 구조 변경)~~ -> 네트워크 두 개를 각각 만들고 train 데이터를 양분하여 학습한 뒤 서로 데이터를 바꾸어 validate. 일정 update step 이후에 total reward개선이 없는 모델은 다시 학습
1. portfolio weight(Action)변화 시각화
1. rolling train
1. 모델 간 비교를 위한 메트릭 정하기
1. 거래 수수료 적용
1. 종목별 상한, 하한 적용
1. ~~batch shuffling~~ DONE


# TODO (장기적)
1. 자연어 처리를 이용한 feature 추가(using fn guide report or sth.)
