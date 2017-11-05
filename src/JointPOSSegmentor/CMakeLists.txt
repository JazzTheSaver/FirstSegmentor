
include_directories(
  model
)
add_executable(JointPOSSegmentor JointPOSSegmentor.cpp)
target_link_libraries(JointPOSSegmentor ${LIBS})
