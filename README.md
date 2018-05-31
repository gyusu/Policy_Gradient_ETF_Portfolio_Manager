# Policy_Gradient_ETF_Portfolio_Manager

Policy Gradient기법을 이용하여
ETF 상품 15개의 포트폴리오를 구성하는 프로그램

# TODO (단기적)
1. loss(critic) 변경 (sharp ratio or mse between portfolio value gain and optimal) 현재는 단순히 portfolio value gain이용중
1. 초기값에 따라 policy 수렴이 달라지는 문제 해결(앙상블 or 모델 구조 변경)
1. portfolio weight(Action)변화 시각화
1. rolling train
1. 모델 간 비교를 위한 메트릭 정하기
1. 거래 수수료 적용
1. 종목별 상한, 하한 적용
1. batch suffling


# TODO (장기적)
1. 자연어 처리를 이용한 feature 추가(using fn guide report or sth.)
